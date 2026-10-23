//! `nucleus serve <config> [--parameters <values>]`: the VM's resident kernel,
//! owned by a service manager.
//!
//! The platform's values file fills the configuration's parameters. It boots
//! every configured program and runs until SIGTERM or SIGINT, then shuts the
//! whole tree down. If a root program exits, the kernel shuts down and exits
//! with an error. The service manager's policy may start it fresh; the local
//! operator composition instead reports the failure without restarting.
//!
//! In the VM, systemd runs it as the `nucleus` user with deploy/vm/nucleus.json
//! and restarts it whenever it exits (deploy/vm/nucleus.service). Every process
//! it runs is a program of that composition or a process one starts: the
//! runtime's own run sandboxed, a project's as the projects' user in no
//! sandbox, in its project's cgroup (crate nucleus-platform). No client has a
//! kernel identity: a window reaches the VM's UI service over a link that
//! service dials to the platform (program-files/ui, module `link`), or an
//! owner-only Unix socket in the local composition.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::{LauncherConfig, Lifetime, boot_entry, start_supervisor};

pub fn serve(config_path: &Path, values: Option<&Path>) -> Result<(), String> {
    let config = LauncherConfig::load_with(config_path, values, crate::write_parameter_event)?;
    if config.lifetime != Lifetime::Resident {
        return Err("nucleus serve requires lifetime=resident".into());
    }
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, Arc::clone(&stop)).map_err(|e| format!("signal handler: {e}"))?;
    }
    let mut supervisor = start_supervisor(&config, Arc::new(crate::write_console_event))?;
    let mut roots = Vec::new();
    let outcome = (|| {
        for entry in &config.boot {
            roots.push((entry.program.clone(), boot_entry(&supervisor, entry, config.ready_timeout_ms)?));
        }
        nucleus_platform::diagnostics::emit(format_args!("nucleus: serving {}", config_path.display()));
        loop {
            if stop.load(Ordering::Acquire) {
                return Ok(());
            }
            if let Some((program, _)) = roots.iter().find(|(_, process)| !supervisor.is_live(*process)) {
                return Err(format!("{program} exited; shutting down for a clean restart"));
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    })();
    let report = supervisor.shutdown();
    outcome?;
    if report.is_clean() { Ok(()) } else { Err(format!("shutdown was not clean: {report:?}")) }
}
