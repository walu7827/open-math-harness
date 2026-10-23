//! Memory service: provider of `nucleus-memory-interface`.
//!
//! It runs the two passes of the write pipeline and serves the read path.
//!
//! ```text
//! remember  -> (the session goes quiet) -> extract: conversation history -> model -> a row
//! a row     -> select -> sync the workspace -> diff -> a confined agent edits MEMORY.md
//! index     -> MEMORY.md, framed as data
//! used      -> the citations of one session, counted against the rows they name
//! ```
//!
//! Why this shape. Extraction is per session, so it is one job per session.
//! Consolidation touches the shared files, so it is one job with one key. This
//! service is the only worker and runs one job at a time; a job it was running
//! when it stopped runs again when it starts, and a failed job waits until its
//! source changes. The agent that does it
//! is a child with read-only candidate inputs, write authority on exactly two
//! output artifacts and a model budget. It has no host filesystem or network
//! grants and cannot reach the baseline, which is in the job store. Publication
//! waits for a successful host reap and validated artifacts; readers see the
//! baselined index.
//!
//! This process does the extraction call itself rather than starting an agent
//! for it, because extraction is one call with one answer. A dispatch that
//! arrives while that call is out is served when it returns, which is why the
//! call has a deadline (`--extract-deadline-ms`) rather than none.
//! The deployment may set `--agent-model-retries` for consolidation calls;
//! zero prevents a failed provider call from being attempted again by that
//! child. Omission preserves the chat agent's retry policy. This does not
//! change the job store's recorded recovery behavior or undo a model charge.
//!
//! Each project remembers apart: its workspace is `<--root>/<project>` and
//! its job store `<--db>/<project>.db`, and a session's extraction, a
//! project's consolidation and what an `index` returns use its project's
//! alone. The service still runs one job at a time across projects.
//!
//! `forget` on `[project]` stops the project's consolidation, drops what
//! waited, closes its job store and removes it with the workspace. The
//! service is the stores' only worker and serves one thing at a time, so no
//! extraction of the project is under way meanwhile; the stopped agent's
//! writes land nowhere, since the `fs` service makes no directory a write
//! needs.
//! `stop` keeps the workspace and extracted memories, but cancels queued
//! work durably and stops consolidation. Restoring the project or restarting
//! this service cannot resume it; new source activity may enqueue new work.
//!
//! Two configured values name one directory: `--root`, the workspaces on this
//! host, which this service diffs and commits; and the resource of the `fs`
//! capability it holds, which is what the consolidating agent edits (a
//! project's under it). If those were two directories the agent's work would
//! land where this service never looks, so before a project's first
//! consolidation the two are checked against each other, through a file only
//! a caller of both could read.
//!
//! Privileges and compromise: it holds `conversation history` for the sessions
//! it covers, a model budget, `fs` on the memory workspace, and authority to
//! start the consolidating agent. A compromised instance can read every
//! conversation in its scope, spend the model budget, rewrite what the system
//! remembers, and put text of its choosing in front of every future session.
//! It holds no credentials and no network.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use nucleus_conversation_interface as conversation;
use nucleus_memories_write::storage::{CONSOLIDATE, CONSOLIDATE_KEY, EXTRACT};
use nucleus_memories_write::{Claim, Outcome, Stage1Output, Store, prompts, workspace};
use nucleus_memory_interface as memory;
use nucleus_model_interface as model;
use nucleus_projects_interface::valid_name;
use nucleus_protocol::{
    ACTION_PROVIDE, ACTION_SPAWN_EXECUTABLE, ActionName, ChildProgram, DispatchId, EndpointName, Error,
    ErrorCode, Grant, HandleId, HandleInfo, KERNEL_ENDPOINT, Payload, ProcessId, ResourcePath,
};
use nucleus_sdk::{Connection, Event};
use serde_json::{Value, json};

/// How often the service looks at its own clock: for sessions that have gone
/// quiet, or a consolidation that has run too long.
const TICK: Duration = Duration::from_millis(2_000);
/// Records one extraction reads from one session, whatever their size.
const MAX_SESSION_RECORDS: usize = 20_000;
/// What the consolidating agent is allowed: enough steps to read the diff and
/// rewrite the index, and a model budget that cannot outlive the job.
const CONSOLIDATE_MAX_STEPS: u32 = 40;
const CONSOLIDATE_MODEL_CALLS: u64 = 48;
/// Room kept for the framing around the remembered text in one `index`.
const INDEX_FRAMING_RESERVE: usize = 4_096;
/// Actions the consolidating agent needs on the workspace. `remove` is in it:
/// dropping a memory that is no longer true is the job, and in this one
/// directory removal is not irreversible -- the baseline holds the
/// previous state and a failed pass is reset to it.
const AGENT_FS_ACTIONS: [&str; 5] = ["read", "list", "create", "append", "remove"];
/// The file the root check writes and reads back. Inside the workspace, and
/// removed straight away, so it is never part of a diff.
const ROOT_CHECK: &str = ".root-check";

fn read_history_pages(
    mut fetch: impl FnMut(Option<u64>) -> Result<conversation::HistoryResponse, String>,
) -> Result<Vec<Value>, String> {
    let mut records = Vec::new();
    let mut after = None;
    loop {
        let response = fetch(after)?;
        let next = response.continuation(after).map_err(|error| error.to_string())?;
        records.extend(response.records);
        if next.is_none() || records.len() >= MAX_SESSION_RECORDS {
            return Ok(records);
        }
        after = next;
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Options {
    /// The projects' memory workspaces on this host, one directory each:
    /// plain files whose baseline (the state of the previous successful
    /// consolidation) is in the project's job store.
    pub root: PathBuf,
    /// Project folders, when execution must be suspended for archived projects.
    pub homes: Option<PathBuf>,
    /// The job stores, one `<project>.db` each, with their baselines.
    /// Defaults to `memory-jobs` beside the workspaces, never inside them,
    /// where the consolidating agent's files are.
    pub db: Option<PathBuf>,
    /// The model that reads a session and writes down what it was worth.
    pub model: String,
    /// The program the consolidating agent is started from.
    pub agent_executable: String,
    /// The model that agent runs on. Defaults to `--model`; either way the
    /// kernel refuses the delegation if it is outside what this service holds.
    pub agent_model: Option<String>,
    /// An explicit deployment override of the consolidating agent's retry
    /// policy. None leaves its default in that agent, the single source.
    pub agent_model_retries: Option<u32>,
    /// How long a session must be quiet before it is read back. A session is
    /// extracted when it stops changing, not while it is being written.
    pub quiet_ms: i64,
    pub extract_deadline_ms: u64,
    /// How long the consolidating agent may run before it is stopped and its
    /// work thrown away.
    pub consolidate_deadline_ms: u64,
    /// Most rows one consolidation works from.
    pub select: usize,
    /// A memory nothing has cited for this long stops being selected.
    pub max_unused_days: i64,
    /// Most of one session's record that reaches the extraction model.
    pub max_session_bytes: usize,
    pub endpoint: EndpointName,
    /// Filesystem provider for the memory workspace, independent of its host.
    pub fs_endpoint: EndpointName,
}

impl Options {
    /// Parses `--root <dir> --model <id> --agent-executable <path>
    /// [--db <dir>] [--projects <dir>] [--agent-model <id>] [--agent-model-retries <n>]
    /// [--quiet-ms <n>]
    /// [--extract-deadline-ms <n>] [--consolidate-deadline-ms <n>]
    /// [--select <n>] [--max-unused-days <n>] [--max-session-bytes <n>]
    /// [--endpoint <name>] [--fs-endpoint <name>]`.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let mut root = None;
        let mut homes = None;
        let mut db = None;
        let mut model = None;
        let mut agent_executable = None;
        let mut agent_model = None;
        let mut agent_model_retries = None;
        let mut quiet_ms = 300_000;
        let mut extract_deadline_ms = 180_000;
        let mut consolidate_deadline_ms = 900_000;
        let mut select = 64;
        let mut max_unused_days = 30;
        let mut max_session_bytes = 120_000;
        let mut endpoint = memory::ENDPOINT.to_owned();
        let mut fs_endpoint = "fs".to_owned();
        let number = |flag: &str, value: &str| {
            value
                .parse::<i64>()
                .map_err(|_| format!("{flag} must be a number"))
                .and_then(|n| if n >= 0 { Ok(n) } else { Err(format!("{flag} must not be negative")) })
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--root" => root = Some(PathBuf::from(value)),
                "--projects" => homes = Some(PathBuf::from(value)),
                "--db" => db = Some(PathBuf::from(value)),
                "--model" => model = Some(value),
                // A path in this process's namespace; a relative name is rooted there.
                "--agent-executable" => {
                    agent_executable = Some(if value.starts_with('/') { value } else { format!("/{value}") })
                }
                "--agent-model" => agent_model = Some(value),
                "--agent-model-retries" => {
                    agent_model_retries = Some(
                        value.parse::<u32>().map_err(|_| format!("{flag} must be an unsigned 32-bit integer"))?,
                    )
                }
                "--quiet-ms" => quiet_ms = number(&flag, &value)?,
                "--extract-deadline-ms" => extract_deadline_ms = number(&flag, &value)? as u64,
                "--consolidate-deadline-ms" => consolidate_deadline_ms = number(&flag, &value)? as u64,
                "--select" => select = number(&flag, &value)? as usize,
                "--max-unused-days" => max_unused_days = number(&flag, &value)?,
                "--max-session-bytes" => max_session_bytes = number(&flag, &value)? as usize,
                "--endpoint" => endpoint = value,
                "--fs-endpoint" => fs_endpoint = value,
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        let options = Self {
            root: root.ok_or("--root is required")?,
            db,
            model: model.ok_or("--model is required")?,
            agent_executable: agent_executable.ok_or("--agent-executable is required")?,
            agent_model,
            agent_model_retries,
            homes,
            quiet_ms,
            extract_deadline_ms,
            consolidate_deadline_ms,
            select,
            max_unused_days,
            max_session_bytes,
            endpoint: EndpointName::new(endpoint).map_err(|e| e.to_string())?,
            fs_endpoint: EndpointName::new(fs_endpoint).map_err(|e| e.to_string())?,
        };
        options.db_path()?;
        if options.homes.as_ref().is_some_and(|homes| !homes.is_dir()) {
            return Err("--projects must name a directory".to_owned());
        }
        Ok(options)
    }

    /// Where job state lives. It must not be inside the workspaces, the
    /// directories the consolidating agent's inputs are copied from.
    pub fn db_path(&self) -> Result<PathBuf, String> {
        let path = match &self.db {
            Some(path) => path.clone(),
            None => self
                .root
                .parent()
                .ok_or("--root has no parent directory; give --db explicitly")?
                .join("memory-jobs"),
        };
        if path.starts_with(&self.root) {
            return Err(format!(
                "the job store {} is inside the workspace {}",
                path.display(),
                self.root.display()
            ));
        }
        Ok(path)
    }

    fn agent_model(&self) -> &str {
        self.agent_model.as_deref().unwrap_or(&self.model)
    }
}

/// The handles this service was started with, checked once.
struct Authority {
    provide: HandleId,
    /// Extraction calls through it; the consolidating agent is delegated a
    /// narrowed copy of it.
    model: HandleId,
    conversation: HandleId,
    fs: HandleId,
    /// The workspace as the `fs` service names it: what the agent is granted,
    /// and the other half of the root check.
    fs_resource: ResourcePath,
    spawn: HandleId,
}

impl Authority {
    fn from(handles: &[HandleInfo], fs_endpoint: &EndpointName) -> Result<Self, String> {
        let find = |endpoint: &str, action: &str, delegable: bool| {
            handles
                .iter()
                .find(|h| {
                    h.endpoint.as_str() == endpoint
                        && h.actions.iter().any(|a| a.as_str() == action)
                        && (!delegable || h.delegable)
                })
                .ok_or_else(|| {
                    let kind = if delegable { "delegable " } else { "" };
                    format!("started without a {kind}`{endpoint}` capability for `{action}`")
                })
        };
        // Delegable, because the consolidating agent runs on them.
        let model = find(model::ENDPOINT, model::ACTION_COMPLETE, true)?;
        let fs = find(fs_endpoint.as_str(), "create", true)?;
        let provide = find(KERNEL_ENDPOINT, ACTION_PROVIDE, false)?.handle;
        let history = find(conversation::ENDPOINT, conversation::ACTION_HISTORY, false)?.handle;
        let spawn = find(KERNEL_ENDPOINT, ACTION_SPAWN_EXECUTABLE, false)?.handle;
        let missing: Vec<&str> = AGENT_FS_ACTIONS
            .iter()
            .copied()
            .filter(|a| !fs.actions.iter().any(|held| held.as_str() == *a))
            .collect();
        if !missing.is_empty() {
            // Not a nicety: without `remove` the agent cannot drop a memory
            // that is no longer true, and without `read` it cannot see the
            // diff. An agent missing one of these would look like one that
            // decided not to bother.
            return Err(format!("the `fs` capability is missing {missing:?}, which consolidation needs"));
        }
        Ok(Self {
            provide,
            model: model.handle,
            conversation: history,
            fs: fs.handle,
            fs_resource: fs.resource.clone(),
            spawn,
        })
    }
}

/// The consolidation that is out: the agent doing it, and what it was given.
struct Running {
    claim: Claim,
    /// The rows it is merging. Marked consumed when it succeeds, never before.
    selected: Vec<Stage1Output>,
    child: ProcessId,
    started: Instant,
    candidate: PathBuf,
    stopping: Option<String>,
}

type Kernel = Connection<std::io::Stdin, std::io::Stdout>;

/// One project's memory: its workspace, its job store, what was last published.
struct Project {
    root: PathBuf,
    store: Store,
    workspace: workspace::Workspace,
    /// An extraction produced a row and consolidation has not been asked for
    /// it yet -- because the job was running when it was produced.
    consolidate_pending: bool,
    /// The workspace and the `fs` capability have been checked against each
    /// other in this process.
    root_checked: bool,
    published: String,
    published_at: u64,
}

impl Project {
    /// Opens `name`'s memory, creating it when it is new; whatever it was
    /// running when this service stopped runs again.
    fn open(options: &Options, name: &str) -> Result<Self, String> {
        let root = options.root.join(name);
        let db_dir = options.db_path()?;
        let db = db_dir.join(format!("{name}.db"));
        if root.join(workspace::INDEX).exists() && !db.exists() {
            return Err(format!("the memory workspace of {name} has no job database at {}", db.display()));
        }
        std::fs::create_dir_all(&root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
        std::fs::create_dir_all(&db_dir).map_err(|e| format!("cannot create {}: {e}", db_dir.display()))?;
        let store = Store::open(&db).map_err(|e| format!("cannot open {}: {e}", db.display()))?;
        store.requeue_interrupted().map_err(|e| e.to_string())?;
        match std::fs::remove_dir_all(root.join(workspace::CANDIDATES)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(format!("cannot remove abandoned candidates: {e}"));
            }
            _ => {}
        }
        let workspace = workspace::Workspace::new(root.clone());
        workspace.init(&store, unix_ms()).map_err(|e| format!("cannot prepare the workspace: {e}"))?;
        let (published, published_at) = store.published().map_err(|e| e.to_string())?;
        Ok(Self {
            root,
            store,
            workspace,
            consolidate_pending: false,
            root_checked: false,
            published,
            published_at: u64::try_from(published_at).unwrap_or(0),
        })
    }
}

pub struct Service<'a> {
    kernel: Kernel,
    options: &'a Options,
    authority: Authority,
    /// The projects' memories, opened at start or when first used.
    projects: BTreeMap<String, Project>,
    /// Sessions that said they moved on, by project, and when they last said
    /// it. A session leaves this when its extraction job is ready to run.
    /// An agent says so after every answered turn, because nothing tells this
    /// service that a session ended: an agent's exit is no message to it. The
    /// map is in memory only, so a restart forgets the sessions still waiting
    /// out their quiet period; such a session is read back only once it says
    /// so again.
    touched: BTreeMap<(String, String), i64>,
    /// The project whose consolidation is out, and it.
    consolidating: Option<(String, Running)>,
    stops: Vec<DispatchId>,
}

fn unix_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| i64::try_from(d.as_millis()).unwrap_or(0))
}

fn failed(what: &str, e: &dyn std::fmt::Display) -> Error {
    Error::new(ErrorCode::ServiceError, format!("{what}: {e}"))
}

/// The project and name of the session `resource` names: `[project, session]`.
fn session_key(resource: &ResourcePath) -> Result<(String, String), Error> {
    match resource.segments() {
        [project, name] if valid_name(project) => Ok((project.clone(), name.clone())),
        _ => Err(Error::new(
            ErrorCode::InvalidInput,
            "the resource path must be a project and one session name: [project, session]",
        )),
    }
}

/// One line per record of a session, in the order they happened.
///
/// Bookkeeping records are left out: what the extraction model is reading is a
/// conversation, and a model call's token counts are not part of one. An
/// unrecognised kind is kept rather than dropped, because a kind added later
/// is more likely to be content than not.
fn lines_of(records: &[Value]) -> Vec<String> {
    let text = |record: &Value, field: &str| record[field].as_str().unwrap_or("").to_owned();
    let mut lines = Vec::new();
    for record in records {
        let kind = record["kind"].as_str().unwrap_or("");
        let line = match kind {
            "user" => format!("user: {}", text(record, "text")),
            "assistant" => format!("assistant: {}", text(record, "text")),
            // Provisional pieces of an answer whose final form is elsewhere in
            // the log, and turn bookkeeping.
            "assistant_delta" | "model_call" | "turn_complete" | "delivered" => continue,
            "tool_call" => {
                format!("tool call {}: {}", text(record, "name"), record["arguments"])
            }
            "tool_result" => {
                let code = record["error_code"].as_str().map_or(String::new(), |c| format!(" [{c}]"));
                format!("tool result{code}: {}", text(record, "summary"))
            }
            "compacted" => format!(
                "[the conversation was summarised here: {} messages, {} tokens]",
                record["messages"], record["tokens"]
            ),
            "error" => format!("error: {}", text(record, "text")),
            other => format!("{other}: {record}"),
        };
        lines.push(line);
    }
    lines
}

/// The session as the extraction model sees it, within `max_bytes`.
///
/// When it does not fit, the middle goes and both ends stay: the start holds
/// what the person asked for and the end holds how it turned out, and a
/// summary missing either is a summary of something else. The cut says how
/// much went, so nothing is dropped silently.
fn fit(lines: &[String], max_bytes: usize) -> String {
    let size = |line: &String| line.len() + 1;
    if lines.iter().map(size).sum::<usize>() <= max_bytes {
        return lines.join("\n");
    }
    let (mut head, mut tail) = (Vec::new(), Vec::new());
    let (mut head_bytes, mut tail_bytes) = (0usize, 0usize);
    let head_budget = max_bytes / 3;
    for line in lines {
        if head_bytes + size(line) > head_budget {
            break;
        }
        head_bytes += size(line);
        head.push(line.clone());
    }
    for line in lines[head.len()..].iter().rev() {
        if tail_bytes + size(line) > max_bytes - head_bytes {
            break;
        }
        tail_bytes += size(line);
        tail.push(line.clone());
    }
    tail.reverse();
    let dropped = lines.len() - head.len() - tail.len();
    head.push(format!("\n[{dropped} records omitted here because the session did not fit]\n"));
    head.extend(tail);
    head.join("\n")
}

impl Service<'_> {
    fn active(&self, name: &str) -> bool {
        self.options
            .homes
            .as_ref()
            .is_none_or(|homes| std::fs::symlink_metadata(homes.join(name)).is_ok_and(|meta| meta.is_dir()))
    }

    fn stop_pending(&mut self, name: &str) -> Result<(), Error> {
        self.touched.retain(|(project, _), _| project != name);
        if let Some(project) = self.projects.get_mut(name) {
            project.store.cancel_pending(unix_ms()).map_err(|e| failed("stop pending memory work", &e))?;
            project.consolidate_pending = false;
        }
        Ok(())
    }

    /// `name`'s memory, opened if it was not.
    fn project(&mut self, name: &str) -> Result<&mut Project, Error> {
        if !self.projects.contains_key(name) {
            let opened =
                Project::open(self.options, name).map_err(|e| failed("open the project's memory", &e))?;
            self.projects.insert(name.to_owned(), opened);
        }
        self.projects.get_mut(name).ok_or_else(|| Error::new(ErrorCode::ServiceError, "the memory vanished"))
    }

    /// What one `index` returns: the framing plus the project's index file, bounded.
    fn index_text(project: &Project) -> String {
        let remembered = project.published.clone();
        let budget = memory::MAX_INDEX_BYTES.saturating_sub(INDEX_FRAMING_RESERVE);
        let remembered = if remembered.len() <= budget {
            remembered
        } else {
            // On a line, and by byte position, because a newline byte is
            // always a character boundary and an arbitrary one is not.
            let cut = remembered.as_bytes()[..budget].iter().rposition(|b| *b == b'\n').unwrap_or(0);
            format!(
                "{}\n[the index is longer than one session may be given and has been cut here]\n",
                &remembered[..cut]
            )
        };
        nucleus_memories_read::injection(&remembered).unwrap_or_default()
    }

    /// Counts one session's use of the memories it named.
    ///
    /// A citation is matched by file name against the rows a successful
    /// consolidation has consumed ([`Store::selected`]), which include every
    /// memory a session can have been shown. Anything else is dropped: a count
    /// decides what survives, and a path nobody wrote must not be able to keep
    /// a memory alive. For the same reason the count is per citing session,
    /// not per report: a conversation cites what it is using every turn.
    fn record_used(project: &Project, citing: &str, paths: &[String], now: i64) -> Result<(), Error> {
        let selected = project.store.selected().map_err(|e| failed("read the selected memories", &e))?;
        let by_file: BTreeMap<String, String> =
            selected.iter().map(|row| (workspace::file_name_for(row), row.session_id.clone())).collect();
        let mut counted = BTreeSet::new();
        for path in paths {
            let name = path.rsplit(['/', '\\']).next().unwrap_or(path);
            if let Some(session) = by_file.get(name) {
                counted.insert(session.clone());
            }
        }
        for session in &counted {
            project.store.record_usage(session, citing, now).map_err(|e| failed("count a citation", &e))?;
        }
        Ok(())
    }

    /// What a person reads: a project's file itself, and what the service is
    /// doing with it. A project with no memory yet is told so; nothing is made
    /// for it.
    fn contents(&mut self, name: &str) -> Result<memory::ContentsResponse, Error> {
        let waiting = self.touched.keys().filter(|(project, _)| project == name).count();
        let consolidating = self.consolidating.as_ref().is_some_and(|(project, _)| project == name);
        if !self.projects.contains_key(name) && !self.options.root.join(name).is_dir() {
            let nothing = memory::ContentsResponse {
                index: String::new(),
                updated_at_ms: 0,
                memories: 0,
                waiting: waiting as u64,
                consolidating,
            };
            return Ok(nothing);
        }
        let project = self.project(name)?;
        let memories = project.store.selected().map_err(|e| failed("count the memories", &e))?.len();
        Ok(memory::ContentsResponse {
            index: project.published.clone(),
            updated_at_ms: project.published_at,
            memories: memories as u64,
            waiting: waiting as u64,
            consolidating,
        })
    }

    /// `forget` (module doc): nothing of `name`'s memory is left.
    fn forget(&mut self, name: &str) -> Result<(), Error> {
        if let Some((_, running)) = self.consolidating.take_if(|(project, _)| project == name) {
            // Not a child of this process any more: gone already, as good as stopped.
            if let Err(e) =
                self.kernel.kill(running.child).map_err(|e| failed("stop the consolidation", &e))?
                && e.code != ErrorCode::PermissionDenied
            {
                self.consolidating = Some((name.to_owned(), running));
                return Err(failed("stop the consolidation", &e.message));
            }
        }
        self.touched.retain(|(project, _), _| project != name);
        // The job store closes with it.
        self.projects.remove(name);
        let db = self.options.db_path().map_err(|e| failed("the job stores", &e))?;
        let mut gone = vec![self.options.root.join(name)];
        gone.extend(["db", "db-wal", "db-shm", "db-journal"].map(|end| db.join(format!("{name}.{end}"))));
        for path in gone {
            let removed = match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_dir() => std::fs::remove_dir_all(&path),
                Ok(_) => std::fs::remove_file(&path),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
                Err(e) => Err(e),
            };
            removed.map_err(|e| failed(&format!("remove {}", path.display()), &e))?;
        }
        Ok(())
    }

    fn serve(&mut self, action: &str, resource: &ResourcePath, payload: &Payload) -> Result<Payload, Error> {
        if action == memory::ACTION_FORGET {
            payload.decode::<nucleus_projects_interface::Empty>()?;
            let [project] = resource.segments() else {
                return Err(Error::new(ErrorCode::InvalidInput, "`forget` is called on a project's path"));
            };
            if !valid_name(project) {
                return Err(Error::new(ErrorCode::InvalidInput, format!("{project:?} is not a project")));
            }
            let project = project.clone();
            self.forget(&project)?;
            return Payload::encode(&json!({}));
        }
        // The one action that is not about a session: a capability scoped to
        // one cannot reach it, which is the point.
        if action == memory::ACTION_CONTENTS {
            payload.decode::<Value>()?;
            let [project] = resource.segments() else {
                return Err(Error::new(
                    ErrorCode::InvalidInput,
                    "`contents` is called on a project's path: what is remembered is not one session's",
                ));
            };
            if !valid_name(project) {
                return Err(Error::new(ErrorCode::InvalidInput, format!("{project:?} is not a project")));
            }
            let project = project.clone();
            return Payload::encode(&self.contents(&project)?);
        }
        let (name, session) = session_key(resource)?;
        match action {
            memory::ACTION_INDEX => {
                payload.decode::<Value>()?;
                let text = Self::index_text(self.project(&name)?);
                Payload::encode(&memory::IndexResponse { text })
            }
            memory::ACTION_USED => {
                let request: memory::UsedRequest = payload.decode()?;
                if request.paths.len() > memory::MAX_USED_PATHS {
                    return Err(Error::new(ErrorCode::InvalidInput, "too many paths in one report"));
                }
                Self::record_used(self.project(&name)?, &session, &request.paths, unix_ms())?;
                Payload::encode(&json!({}))
            }
            memory::ACTION_REMEMBER => {
                payload.decode::<memory::RememberRequest>()?;
                if !self.active(&name) {
                    return Err(Error::new(ErrorCode::Unavailable, "the project is not active"));
                }
                self.project(&name)?;
                self.touched.insert((name, session), unix_ms());
                Payload::encode(&json!({}))
            }
            _ => Err(Error::new(ErrorCode::InvalidInput, "unknown action")),
        }
    }

    /// One session's record, in as many calls as the provider's page size takes.
    fn read_session(&mut self, project: &str, session: &str) -> Result<Vec<Value>, String> {
        let resource = ResourcePath::new([project, session]).map_err(|e| e.to_string())?;
        read_history_pages(|after| {
            let request = conversation::HistoryRequest { after_seq: after };
            let reply = self
                .kernel
                .call(self.authority.conversation, conversation::ACTION_HISTORY, &resource, &request, None)
                .map_err(|e| e.to_string())?;
            reply
                .and_then(|payload| payload.decode::<conversation::HistoryResponse>())
                .map_err(|e| format!("{:?}: {}", e.code, e.message))
        })
    }

    /// Reads one session back and records what it was worth remembering.
    ///
    /// The model call blocks this service for as long as its deadline allows.
    /// A reply that does not parse is a failure, not an empty memory: the
    /// difference decides whether the session is read again.
    fn extract(&mut self, name: &str, claim: &Claim) -> Result<(), String> {
        let session = claim.job_key.clone();
        let now = unix_ms();
        let records = match self.read_session(name, &session) {
            Ok(records) => records,
            Err(e) => return self.end(name, claim, &Outcome::Failed(e)),
        };
        if records.is_empty() {
            return self.end(name, claim, &Outcome::SucceededNoOutput);
        }
        let at = |record: Option<&Value>| record.and_then(|r| r["at_ms"].as_i64()).unwrap_or(now);
        let contents = fit(&lines_of(&records), self.options.max_session_bytes);
        let input = prompts::extract_input(&prompts::SessionInput {
            session_id: &session,
            started_at: &format!("{} (unix milliseconds)", at(records.first())),
            contents: &contents,
        });
        let request = model::CompletionRequest {
            messages: vec![
                model::Message::text(model::Role::System, prompts::EXTRACT_SYSTEM.to_owned()),
                model::Message::text(model::Role::User, input),
            ],
            tools: Vec::new(),
            max_output_tokens: 0,
            temperature: None,
            reasoning_effort: None,
            tool_choice: None,
            project: Some(name.to_owned()),
        };
        let resource = match model::resource_for(&self.options.model) {
            Ok(resource) => resource,
            Err(e) => return Err(e.to_string()),
        };
        let reply = self
            .kernel
            .call(
                self.authority.model,
                model::ACTION_COMPLETE,
                &resource,
                &request,
                Some(self.options.extract_deadline_ms),
            )
            .map_err(|e| e.to_string())?;
        let answer = match reply.and_then(|payload| payload.decode::<model::CompletionResponse>()) {
            Ok(response) => response.message.content,
            Err(e) => {
                let error = format!("the extraction call failed ({:?}): {}", e.code, e.message);
                return self.end(name, claim, &Outcome::Failed(error));
            }
        };
        let extracted = match prompts::Extracted::parse(&answer) {
            Ok(extracted) => extracted,
            Err(e) => {
                let error = format!("the extraction reply did not parse: {e}");
                return self.end(name, claim, &Outcome::Failed(error));
            }
        };
        if extracted.is_empty() {
            return self.end(name, claim, &Outcome::SucceededNoOutput);
        }
        let slug = extracted.session_slug.trim();
        let row = Stage1Output {
            session_id: session,
            source_updated_at: at(records.last()),
            // The reply as it came, which is the evidence for what was kept.
            raw_memory: answer.trim().to_owned(),
            session_summary: extracted.session_summary,
            session_slug: (!slug.is_empty()).then(|| slug.to_owned()),
            generated_at: now,
            usage_count: None,
            last_usage: None,
        };
        let project = self.project(name).map_err(|e| e.message)?;
        project.store.put_extraction(&row).map_err(|e| e.to_string())?;
        project.consolidate_pending = true;
        self.end(name, claim, &Outcome::Succeeded)
    }

    fn end(&mut self, name: &str, claim: &Claim, outcome: &Outcome) -> Result<(), String> {
        let project = self.project(name).map_err(|e| e.message)?;
        project.store.finish(claim, outcome, unix_ms()).map_err(|e| e.to_string())
    }

    /// The `fs` resource of `name`'s workspace: the project's, under the one
    /// this service holds.
    fn fs_resource(&self, name: &str) -> Result<Vec<String>, String> {
        let mut segments = self.authority.fs_resource.segments().to_vec();
        segments.push(name.to_owned());
        Ok(segments)
    }

    /// `--root` and the granted `fs` capability must be one directory.
    ///
    /// Checked through a file written here and read back through the
    /// capability, because that is the only question worth asking: not whether
    /// two strings match, but whether the agent's writes land where this
    /// service looks.
    fn check_root(&mut self, name: &str) -> Result<(), String> {
        let root = {
            let project = self.project(name).map_err(|e| e.message)?;
            if project.root_checked {
                return Ok(());
            }
            project.root.clone()
        };
        let marker = format!("{}\n", unix_ms());
        let file = root.join(ROOT_CHECK);
        std::fs::write(&file, &marker).map_err(|e| format!("cannot write in --root: {e}"))?;
        let mut segments = self.fs_resource(name)?;
        segments.push(ROOT_CHECK.to_owned());
        let path = ResourcePath::new(segments).map_err(|e| e.to_string())?;
        let reply = self.kernel.call(self.authority.fs, "read", &path, &json!({}), None);
        let read = reply.map_err(|e| e.to_string())?.and_then(|payload| payload.decode::<Value>());
        let _ = std::fs::remove_file(&file);
        let content = match &read {
            Ok(value) => value["content"].as_str().unwrap_or_default().to_owned(),
            Err(e) => format!("<unreadable: {:?}: {}>", e.code, e.message),
        };
        if content != marker {
            return Err(format!(
                "--root {} and the `fs` capability for {:?} are not the same directory: what the \
                 consolidating agent writes would land where this service never reads",
                root.display(),
                self.fs_resource(name)?
            ));
        }
        self.project(name).map_err(|e| e.message)?.root_checked = true;
        Ok(())
    }

    /// Puts the selected rows in the workspace and starts the agent that
    /// merges them, unless nothing changed since the last baseline.
    fn start_consolidation(&mut self, name: &str, claim: Claim) -> Result<(), String> {
        let now = unix_ms();
        if let Err(e) = self.check_root(name) {
            return self.end(name, &claim, &Outcome::Failed(e));
        }
        let (select, max_unused_days) = (self.options.select, self.options.max_unused_days);
        let project = self.project(name).map_err(|e| e.message)?;
        let root = project.root.clone();
        let selected = project
            .store
            .select_for_consolidation(select, max_unused_days, now)
            .map_err(|e| e.to_string())?;
        let (ws, store) = (&project.workspace, &project.store);
        let prepared = ws
            .reset(store)
            .and_then(|()| workspace::sync(&root, &selected))
            .and_then(|()| ws.is_dirty(store))
            .and_then(|dirty| if dirty { ws.write_diff(store).map(Some) } else { Ok(None) });
        match prepared {
            Err(e) => {
                ws.reset(store).map_err(|reset| format!("preparation failed: {e}; reset failed: {reset}"))?;
                let error = format!("the workspace could not be prepared: {e}");
                return self.end(name, &claim, &Outcome::Failed(error));
            }
            // Nothing has changed since the last consolidation, so there is
            // nothing to consolidate and no model is called.
            Ok(None) => return self.end(name, &claim, &Outcome::SucceededNoOutput),
            Ok(Some(_)) => {}
        };
        let key = format!("attempt-{now}");
        let candidate = workspace::candidate(&root, &key).map_err(|e| e.to_string())?;
        let mut parts = self.fs_resource(name)?;
        parts.extend([workspace::CANDIDATES.to_owned(), key]);
        let resource = ResourcePath::new(parts).map_err(|e| e.to_string())?;
        let diff = format!("{}/{}", resource.segments().join("/"), workspace::DIFF);
        let grants = self.agent_grants(&resource)?;
        let mut program = ChildProgram {
            executable: self.options.agent_executable.clone(),
            args: vec![
                "--model".to_owned(),
                self.options.agent_model().to_owned(),
                "--system-prompt".to_owned(),
                prompts::consolidate(&diff),
                "--task".to_owned(),
                "Consolidate the memory workspace now, as your instructions describe.".to_owned(),
                "--max-steps".to_owned(),
                CONSOLIDATE_MAX_STEPS.to_string(),
                // The project its model calls are made for: it has no
                // conversation to name one.
                "--project".to_owned(),
                name.to_owned(),
            ],
            binds: Vec::new(),
            network: false,
        };
        if let Some(retries) = self.options.agent_model_retries {
            program.args.extend(["--model-retries".to_owned(), retries.to_string()]);
        }
        let spawned =
            self.kernel.spawn_executable(self.authority.spawn, program, grants).map_err(|e| e.to_string())?;
        match spawned {
            Ok(child) => {
                let running =
                    Running { claim, selected, child, started: Instant::now(), candidate, stopping: None };
                self.consolidating = Some((name.to_owned(), running));
                Ok(())
            }
            Err(e) => {
                let project = self.project(name).map_err(|e| e.message)?;
                project
                    .workspace
                    .reset(&project.store)
                    .map_err(|e| format!("spawn failed and baseline reset failed: {e}"))?;
                std::fs::remove_dir_all(&candidate).map_err(|e| format!("candidate cleanup failed: {e}"))?;
                let error =
                    format!("the consolidating agent could not be started ({:?}): {}", e.code, e.message);
                self.end(name, &claim, &Outcome::Failed(error))
            }
        }
    }

    /// What the consolidating agent gets: the workspace, and a bounded slice
    /// of this service's model budget. Nothing is delegable: it does not get
    /// to start anything of its own.
    fn agent_grants(&self, resource: &ResourcePath) -> Result<Vec<Grant>, String> {
        let actions = |names: &[&str]| {
            names
                .iter()
                .map(|a| ActionName::new(*a))
                .collect::<Result<BTreeSet<_>, _>>()
                .map_err(|e| e.to_string())
        };
        let mut grants = vec![
            Grant {
                from: self.authority.model,
                actions: actions(&[model::ACTION_COMPLETE])?,
                resource: model::resource_for(self.options.agent_model()).map_err(|e| e.to_string())?,
                ttl_ms: None,
                delegable: false,
                max_calls: Some(CONSOLIDATE_MODEL_CALLS),
            },
            Grant {
                from: self.authority.fs,
                actions: actions(&["read", "list"])?,
                resource: resource.clone(),
                ttl_ms: None,
                delegable: false,
                max_calls: None,
            },
        ];
        for name in [workspace::INDEX, workspace::SUMMARY] {
            let mut parts = resource.segments().to_vec();
            parts.push(name.to_owned());
            grants.push(Grant {
                from: self.authority.fs,
                actions: actions(&["create", "append", "remove"])?,
                resource: ResourcePath::new(parts).map_err(|e| e.to_string())?,
                ttl_ms: None,
                delegable: false,
                max_calls: None,
            });
        }
        Ok(grants)
    }

    /// Host exit success is necessary but not sufficient. Only validated
    /// artifacts enter the baseline; readers use that version.
    ///
    /// Publishing is three writes, not one transaction: the new baseline, the
    /// rows marked consolidated, then the job finished. A crash between them
    /// leaves the new baseline published with the job still running in the
    /// store; the job runs again at the next start
    /// ([`Store::requeue_interrupted`]), and nothing else reconciles it.
    fn consolidation_ended(&mut self, child: ProcessId, success: bool) -> Result<(), String> {
        let Some((name, running)) = self.consolidating.take_if(|(_, r)| r.child == child) else {
            return Ok(());
        };
        let project = self.project(&name).map_err(|e| e.message)?;
        let accepted = if let Some(reason) = running.stopping {
            Err(reason)
        } else if !success {
            Err("the consolidating agent did not finish successfully".to_owned())
        } else {
            workspace::accept_candidate(&project.root, &running.candidate)
                .map_err(|e| e.to_string())
                .and_then(|()| {
                    project.workspace.commit_baseline(&project.store, unix_ms()).map_err(|e| e.to_string())
                })
        };
        match accepted {
            Ok(()) => {
                let (published, at) = project.store.published().map_err(|e| e.to_string())?;
                project.store.mark_consolidated(&running.selected).map_err(|e| e.to_string())?;
                project.published = published;
                project.published_at = u64::try_from(at).unwrap_or(0);
                self.end(&name, &running.claim, &Outcome::Succeeded)?;
            }
            Err(error) => {
                project
                    .workspace
                    .reset(&project.store)
                    .map_err(|e| format!("{error}; baseline reset failed: {e}"))?;
                self.end(&name, &running.claim, &Outcome::Failed(error))?;
            }
        }
        std::fs::remove_dir_all(&running.candidate).map_err(|e| format!("candidate cleanup failed: {e}"))?;
        Ok(())
    }

    /// Starts whatever is due, and keeps what is running honest.
    fn advance(&mut self) -> Result<(), String> {
        let now = unix_ms();
        // Sessions that have stopped changing. A session still being written
        // is left alone: extracting it would summarise half a conversation.
        let quiet: Vec<(String, String)> = self
            .touched
            .iter()
            .filter(|(_, last)| now - **last >= self.options.quiet_ms)
            .map(|(key, _)| key.clone())
            .collect();
        for key in quiet {
            let store = &self.project(&key.0).map_err(|e| e.message)?.store;
            store.enqueue(EXTRACT, &key.1).map_err(|e| e.to_string())?;
            // False means it is running: the session stays, and says so again
            // when that run is done.
            if store.reopen(EXTRACT, &key.1).map_err(|e| e.to_string())? {
                self.touched.remove(&key);
            }
        }
        for project in self.projects.values_mut().filter(|p| p.consolidate_pending) {
            project.store.enqueue(CONSOLIDATE, CONSOLIDATE_KEY).map_err(|e| e.to_string())?;
            project.consolidate_pending =
                !project.store.reopen(CONSOLIDATE, CONSOLIDATE_KEY).map_err(|e| e.to_string())?;
        }
        if let Some((_, running)) = &self.consolidating {
            let child = running.child;
            let over =
                running.started.elapsed() > Duration::from_millis(self.options.consolidate_deadline_ms);
            if over && running.stopping.is_none() {
                self.kernel.kill(child).map_err(|e| e.to_string())?.map_err(|e| e.to_string())?;
                if let Some((_, running)) = &mut self.consolidating {
                    running.stopping = Some("the consolidating agent ran past its deadline".to_owned());
                }
            }
            return Ok(());
        }
        // One job at a time, and never an extraction while a consolidation is
        // out: they contend for the same budget, and a project's for the same rows.
        let names: Vec<String> = self.projects.keys().filter(|name| self.active(name)).cloned().collect();
        for name in &names {
            if let Some(claim) = self.projects[name].store.claim(EXTRACT).map_err(|e| e.to_string())? {
                return self.extract(name, &claim);
            }
        }
        for name in &names {
            if let Some(claim) = self.projects[name].store.claim(CONSOLIDATE).map_err(|e| e.to_string())? {
                return self.start_consolidation(name, claim);
            }
        }
        Ok(())
    }

    /// Restores the workspace after the kernel has reaped our descendants.
    /// `Terminate` revokes this process's authority: only local cleanup remains.
    fn shutdown(&mut self) -> Result<(), String> {
        if let Some((name, _)) = self.consolidating.take() {
            let project = self.project(&name).map_err(|e| e.message)?;
            project.workspace.reset(&project.store).map_err(|e| format!("shutdown reset failed: {e}"))?;
            // Not a failure of the pass: it runs again when the service starts.
            project.store.requeue_interrupted().map_err(|e| e.to_string())?;
        }
        Ok(())
    }
}

/// Provides the endpoint, serves it, and runs the pipeline until told to stop.
pub fn run(options: &Options) -> Result<(), String> {
    std::fs::create_dir_all(&options.root).map_err(|e| format!("cannot create --root: {e}"))?;
    // This service is the stores' only worker: whatever a project's was
    // running when it stopped runs again now.
    let mut projects = BTreeMap::new();
    for entry in std::fs::read_dir(&options.root).map_err(|e| format!("cannot list --root: {e}"))? {
        let entry = entry.map_err(|e| format!("cannot list --root: {e}"))?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else { continue };
        if valid_name(&name) && entry.file_type().is_ok_and(|kind| kind.is_dir()) {
            projects.insert(name.clone(), Project::open(options, &name)?);
        }
    }

    let mut kernel = Connection::from_stdio().map_err(|e| e.to_string())?;
    let authority = Authority::from(kernel.handles(), &options.fs_endpoint)?;
    let declared = memory::interface(options.endpoint.clone()).map_err(|e| e.to_string())?;
    kernel.provide(authority.provide, declared).map_err(|e| e.to_string())?.map_err(|e| {
        format!("cannot provide `{}`: {:?}: {}", options.endpoint.as_str(), e.code, e.message)
    })?;

    let mut service = Service {
        kernel,
        options,
        authority,
        projects,
        touched: BTreeMap::new(),
        consolidating: None,
        stops: Vec::new(),
    };
    loop {
        service.advance()?;
        let event = service.kernel.next_event_within(TICK).map_err(|e| e.to_string())?;
        match event {
            None => continue,
            Some(Event::Dispatch(d)) if d.action.as_str() == memory::ACTION_STOP => {
                let result = (|| {
                    d.payload.decode::<nucleus_projects_interface::Empty>()?;
                    let [project] = d.resource.segments() else {
                        return Err(Error::new(ErrorCode::InvalidInput, "stop is called on [project]"));
                    };
                    if !valid_name(project) {
                        return Err(Error::new(ErrorCode::InvalidInput, "invalid project name"));
                    }
                    service.stop_pending(project)?;
                    if let Some((name, running)) = &mut service.consolidating
                        && name == project
                    {
                        if running.stopping.is_none() {
                            service
                                .kernel
                                .kill(running.child)
                                .map_err(|e| failed("stop consolidation", &e))??;
                        }
                        running.stopping = Some("the project was stopped".to_owned());
                        service.stops.push(d.id);
                        Ok(None)
                    } else {
                        Payload::encode(&json!({})).map(Some)
                    }
                })();
                match result {
                    Ok(None) => {}
                    Ok(Some(answer)) => service.kernel.reply(d.id, Ok(answer)).map_err(|e| e.to_string())?,
                    Err(e) => service.kernel.reply(d.id, Err(e)).map_err(|e| e.to_string())?,
                }
            }
            Some(Event::Dispatch(d)) => {
                let result = service.serve(d.action.as_str(), &d.resource, &d.payload);
                service.kernel.reply(d.id, result).map_err(|e| e.to_string())?;
            }
            Some(Event::ChildExited(_)) => {}
            Some(Event::ChildReaped { child, success, .. }) => {
                let stopped =
                    service.consolidating.as_ref().is_some_and(|(_, running)| running.child == child);
                service.consolidation_ended(child, success)?;
                if stopped {
                    for dispatch in service.stops.drain(..) {
                        service
                            .kernel
                            .reply(dispatch, Payload::encode(&json!({})))
                            .map_err(|e| e.to_string())?;
                    }
                }
            }
            Some(Event::CancelDispatch(id)) => service.stops.retain(|waiting| *waiting != id),
            Some(Event::Terminate) => return service.shutdown(),
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
