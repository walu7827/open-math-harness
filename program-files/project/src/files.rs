//! The window's file operations in one working directory of the project:
//! list, download, mkdir, rename and remove (uploads are in `uploads.rs`).
//! Every path is resolved beneath the working directory with no link
//! followed, and the last component is acted on through its opened parent,
//! never by name from the root again.
//!
//! A revision includes the SHA-256 of the opened file's contents: equal-size
//! writes can share both mtime and ctime on a coarse filesystem clock. Each
//! download hashes the whole file, retaining only its requested chunk; no
//! metadata-keyed cache can distinguish those writes. This costs a full read
//! per chunk. A read that observes changed metadata or an early EOF fails.
//! It is still an observation, not a snapshot or a lock against shell writers
//! changing the file during the read or after a save's final check.

use std::os::fd::{AsFd, OwnedFd};

use base64::Engine;
use nucleus_projects_interface::{
    CHUNK_BYTES, Chunk, Entry, Kind, Listing, MAX_FILE_BYTES, PAGE_ENTRIES, segments,
};
use nucleus_protocol::{Error, ErrorCode};
use rustix::fs::{AtFlags, FileType, Mode, OFlags, RenameFlags};
use sha2::{Digest, Sha256};

use crate::home::{Home, errno, fresh_name, open_beneath, open_dir};

fn invalid(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::InvalidInput, message.into())
}

/// The parent of `path` (opened) and its last name. `path` must name
/// something beneath the working directory, not the directory itself.
pub fn parent(dir: impl AsFd, path: &str) -> Result<(OwnedFd, &str), Error> {
    let parts = segments(path)?;
    let Some((name, above)) = parts.split_last() else {
        return Err(invalid("the path names the working directory itself"));
    };
    let above = above.join("/");
    let parent = open_dir(dir, &above).map_err(|e| errno(&format!("{above:?}"), e))?;
    Ok((parent, name))
}

/// Refuses to rename or remove the repository's own `.git`, which would
/// destroy it.
fn protected(path: &str) -> Result<(), Error> {
    if path == ".git" {
        return Err(invalid("the working directory's .git is not removed or renamed here"));
    }
    Ok(())
}

fn millis(stat: &rustix::fs::Stat) -> u64 {
    let seconds = u64::try_from(stat.st_mtime).unwrap_or(0);
    let nanos = stat.st_mtime_nsec;
    seconds.saturating_mul(1000).saturating_add(nanos / 1_000_000)
}

fn metadata_revision(stat: &rustix::fs::Stat) -> String {
    format!(
        "{:x}:{:x}:{:x}:{:x}:{:x}:{:x}:{:x}",
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime,
        stat.st_mtime_nsec,
        stat.st_ctime,
        stat.st_ctime_nsec
    )
}

/// The revision of an already opened regular file. Bound its size again here
/// because a staged upload's descriptor may have been changed by a shell writer
/// before publication. No path is reopened and pread leaves the offset alone.
pub fn revision(fd: impl AsFd, stat: &rustix::fs::Stat) -> Result<String, Error> {
    if !u64::try_from(stat.st_size).is_ok_and(|size| size <= MAX_FILE_BYTES) {
        return Err(invalid(format!("file is larger than {MAX_FILE_BYTES} bytes")));
    }
    observe(fd, stat, 0, 0).map(|(revision, _)| revision)
}

/// Hash the bytes of the file and keep the requested range from that same
/// read. In particular, a download never returns bytes from one read with
/// the content hash of a later read of the same descriptor.
fn observe(
    fd: impl AsFd,
    stat: &rustix::fs::Stat,
    offset: u64,
    want: usize,
) -> Result<(String, Vec<u8>), Error> {
    let changed = || Error::new(ErrorCode::ServiceError, "The file changed while it was read.");
    let size = u64::try_from(stat.st_size).unwrap_or(0);
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut data = Vec::with_capacity(want);
    let mut position = 0;
    while position < size {
        let amount = usize::try_from((size - position).min(buffer.len() as u64)).unwrap_or(buffer.len());
        let read = match rustix::io::pread(&fd, &mut buffer[..amount], position) {
            Ok(0) => return Err(changed()),
            Ok(read) => read,
            Err(rustix::io::Errno::INTR) => continue,
            Err(e) => return Err(errno("reading file content", e)),
        };
        hash.update(&buffer[..read]);
        let start = position.max(offset);
        let end = (position + read as u64).min(offset + want as u64);
        if start < end {
            data.extend_from_slice(&buffer[(start - position) as usize..(end - position) as usize]);
        }
        position += read as u64;
    }
    let after = rustix::fs::fstat(&fd).map_err(|e| errno("reading file metadata", e))?;
    let metadata = metadata_revision(stat);
    if metadata != metadata_revision(&after) {
        return Err(changed());
    }
    Ok((format!("{metadata}:{:x}", hash.finalize()), data))
}

fn kind(mode: u32) -> Kind {
    match FileType::from_raw_mode(mode) {
        FileType::RegularFile => Kind::File,
        FileType::Directory => Kind::Dir,
        FileType::Symlink => Kind::Link,
        _ => Kind::Other,
    }
}

/// Where an entry stands in listing order: directories first, then by name
/// (as shown, a name that is not UTF-8 with U+FFFD), then by the name's bytes.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Place {
    rank: u8,
    shown: String,
    name: Vec<u8>,
}

impl Place {
    /// The cursor a page ends at: opaque to the caller, `<rank>/<name>`.
    fn cursor(&self) -> String {
        format!("{}/{}", self.rank, self.shown)
    }

    /// Whether the entry comes after `cursor`.
    fn after(&self, cursor: &(u8, String)) -> bool {
        (self.rank, &self.shown) > (cursor.0, &cursor.1)
    }
}

fn parse_cursor(cursor: &str) -> Result<(u8, String), Error> {
    match cursor.split_once('/') {
        Some(("0", name)) => Ok((0, name.to_owned())),
        Some(("1", name)) => Ok((1, name.to_owned())),
        _ => Err(invalid(format!("{cursor:?} is not the `next` of a listing"))),
    }
}

/// One page of `path`: the first [`PAGE_ENTRIES`] entries after `after`, in
/// listing order. Each page reads the whole directory anew and keeps only
/// the entries that may still be in it, never more than a page and one, so a
/// directory of any size is listed page by page in bounded memory; pages are
/// observations, not one snapshot.
pub fn list(dir: impl AsFd, path: &str, after: Option<&str>) -> Result<Listing, Error> {
    segments(path)?;
    let after = after.map(parse_cursor).transpose()?;
    let opened = open_dir(dir, path).map_err(|e| errno(&format!("{path:?}"), e))?;
    let reading = |e: rustix::io::Errno| errno(&format!("{path:?}"), e);
    let mut kept: std::collections::BinaryHeap<Place> = std::collections::BinaryHeap::new();
    for entry in rustix::fs::Dir::read_from(&opened).map_err(reading)? {
        let entry = entry.map_err(reading)?;
        let name = entry.file_name().to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        let shown = String::from_utf8_lossy(name).into_owned();
        let directory = match entry.file_type() {
            FileType::Unknown => {
                match rustix::fs::statat(&opened, entry.file_name(), AtFlags::SYMLINK_NOFOLLOW) {
                    Ok(stat) => FileType::from_raw_mode(stat.st_mode) == FileType::Directory,
                    Err(_) => continue, // removed since it was listed
                }
            }
            known => known == FileType::Directory,
        };
        let place = Place { rank: if directory { 0 } else { 1 }, shown, name: name.to_vec() };
        if after.as_ref().is_some_and(|after| !place.after(after)) {
            continue;
        }
        if kept.len() > PAGE_ENTRIES && kept.peek().is_some_and(|last| place >= *last) {
            continue;
        }
        kept.push(place);
        if kept.len() > PAGE_ENTRIES + 1 {
            kept.pop();
        }
    }
    let mut places = kept.into_sorted_vec();
    let more = places.len() > PAGE_ENTRIES;
    places.truncate(PAGE_ENTRIES);
    let next = if more { places.last().map(Place::cursor) } else { None };
    let mut entries = Vec::with_capacity(places.len());
    for place in places {
        let Ok(name) = std::ffi::CString::new(place.name) else { continue };
        let Ok(stat) = rustix::fs::statat(&opened, name.as_c_str(), AtFlags::SYMLINK_NOFOLLOW) else {
            continue; // removed since it was listed
        };
        entries.push(Entry {
            name: place.shown,
            kind: kind(stat.st_mode),
            size: u64::try_from(stat.st_size).unwrap_or(0),
            modified: millis(&stat),
        });
    }
    Ok(Listing { entries, next })
}

pub fn download(dir: impl AsFd, path: &str, offset: u64) -> Result<Chunk, Error> {
    if segments(path)?.is_empty() {
        return Err(invalid("a file path is required"));
    }
    // Non-blocking, so a FIFO planted under the name cannot hang the host.
    let fd = open_beneath(dir, path, OFlags::RDONLY | OFlags::NOFOLLOW | OFlags::NONBLOCK, Mode::empty())
        .map_err(|e| errno(&format!("{path:?}"), e))?;
    let stat = rustix::fs::fstat(&fd).map_err(|e| errno(&format!("{path:?}"), e))?;
    if FileType::from_raw_mode(stat.st_mode) != FileType::RegularFile {
        return Err(invalid(format!("{path:?} is not a regular file")));
    }
    let size = u64::try_from(stat.st_size).unwrap_or(0);
    if size > MAX_FILE_BYTES {
        return Err(invalid(format!("{path:?} is larger than {MAX_FILE_BYTES} bytes")));
    }
    if offset > size {
        return Err(invalid(format!("offset {offset} is past the end of {path:?} ({size} bytes)")));
    }
    let want = usize::try_from((size - offset).min(CHUNK_BYTES as u64)).unwrap_or(CHUNK_BYTES);
    let (revision, data) = observe(&fd, &stat, offset, want)?;
    let end = offset + data.len() as u64;
    Ok(Chunk {
        data_base64: base64::engine::general_purpose::STANDARD.encode(&data),
        size,
        modified: millis(&stat),
        revision,
        next_offset: (end < size).then_some(end),
    })
}

pub fn mkdir(dir: impl AsFd, path: &str) -> Result<(), Error> {
    let (parent, name) = parent(dir, path)?;
    rustix::fs::mkdirat(&parent, name, Mode::from_raw_mode(0o777)).map_err(|e| errno(&format!("{path:?}"), e))
}

pub fn rename(dir: impl AsFd, from: &str, to: &str) -> Result<(), Error> {
    protected(from)?;
    protected(to)?;
    let (from_parent, from_name) = parent(&dir, from)?;
    let (to_parent, to_name) = parent(&dir, to)?;
    rustix::fs::renameat_with(&from_parent, from_name, &to_parent, to_name, RenameFlags::NOREPLACE)
        .map_err(|e| errno(&format!("{from:?} to {to:?}"), e))
}

/// Removes a file, a link (not its target) or a whole tree. The entry is
/// first renamed into the folder's trash, at once and atomically, then deleted
/// there on a thread of its own, so a large tree does not hold the call.
pub fn remove(home: &Home, dir: impl AsFd, path: &str) -> Result<(), Error> {
    protected(path)?;
    let (parent, name) = parent(dir, path)?;
    rustix::fs::statat(&parent, name, AtFlags::SYMLINK_NOFOLLOW)
        .map_err(|e| errno(&format!("{path:?}"), e))?;
    let trash = home.staging("trash")?;
    let doomed = fresh_name("removed");
    rustix::fs::renameat_with(&parent, name, &trash, doomed.as_str(), RenameFlags::NOREPLACE)
        .map_err(|e| errno(&format!("{path:?}"), e))?;
    std::thread::spawn(move || empty_trash(trash, &doomed));
    Ok(())
}

/// Deletes one entry of the trash. Through the trash directory's own
/// descriptor, so whatever replaces its name meanwhile is not what goes.
fn empty_trash(trash: OwnedFd, name: &str) {
    let trash = cap_std::fs::Dir::from_std_file(std::fs::File::from(trash));
    let removed = match trash.symlink_metadata(name) {
        Ok(meta) if meta.is_dir() => trash.remove_dir_all(name),
        Ok(_) => trash.remove_file(name),
        Err(e) => Err(e),
    };
    if let Err(e) = removed {
        eprintln!("nucleus-project: a removed entry stays in the trash: {e}");
    }
}

/// Deletes what an earlier run left in the trash and the upload and bundle
/// staging areas: fail and restart clean.
pub fn clear_staging(home: &Home) {
    for area in ["trash", "uploads", crate::bundles::AREA] {
        let Ok(fd) = home.staging(area) else { continue };
        let dir = cap_std::fs::Dir::from_std_file(std::fs::File::from(fd));
        let Ok(entries) = dir.entries() else { continue };
        for entry in entries.flatten() {
            let name = entry.file_name();
            let _ = match entry.file_type() {
                Ok(kind) if kind.is_dir() => dir.remove_dir_all(&name),
                _ => dir.remove_file(&name),
            };
        }
    }
}
