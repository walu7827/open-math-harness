//! The memory loop end to end with real confined processes: a session is read
//! back after it goes quiet, a confined agent consolidates what was extracted,
//! and the next session starts knowing it; a deleted project's memory is
//! forgotten whole.
//!
//! The model is scripted per request, keyed on text only that request carries:
//! the extraction prompt, the consolidation brief, or neither (an ordinary
//! chat turn).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::time::{Duration, Instant};

use nucleus_e2e::http::{FakeHttp, Scripted, completion, tool_request};
use nucleus_e2e::{Scenario, TIMEOUT, cap, program};
use nucleus_kernel::CapSpec;
use nucleus_platform::Supervisor;
use serde_json::json;

const MODEL: &str = "deepseek/deepseek-v4-flash";
const KEY: &str = "sk-or-test-MEMORY-0123456789";
/// A line only the consolidating agent writes, so finding it in a later
/// request proves it came out of the memory the pipeline built.
const REMEMBERED: &str = "the owner asks for the plan before the edit";

fn memory_grants() -> Vec<CapSpec> {
    let agent = "nucleus-chat-agent".to_owned();
    let mut own = vec![
        cap("kernel", &["provide"], &["endpoint", "memory"]),
        cap("kernel", &["spawn_executable"], &["executable", "programs", &agent]),
        cap("conversation", &["history"], &[]),
    ];
    for grant in &mut own {
        grant.delegable = false;
    }
    own.push(cap("model", &["complete"], &["deepseek"]));
    own.push(cap("fs", &["read", "list", "create", "append", "remove"], &["memories"]));
    own
}

/// The sessions service's authority, including the memory endpoint it narrows to
/// each session.
fn sessions_grants() -> Vec<CapSpec> {
    let mut provide = cap("kernel", &["provide"], &["endpoint", "sessions"]);
    let mut catalog = cap("conversation", &["list"], &[]);
    provide.delegable = false;
    catalog.delegable = false;
    vec![
        provide,
        nucleus_e2e::session_spawn(),
        catalog,
        cap("conversation", &["receive", "post", "history"], &[]),
        cap("model", &["complete"], &["deepseek"]),
        cap("memory", &["index", "used", "remember"], &[]),
    ]
}

fn ui_grants() -> Vec<CapSpec> {
    let mut grants = vec![
        cap("conversation", &["say", "history"], &[]),
        cap("sessions", &["create", "stop", "list"], &[]),
    ];
    for grant in &mut grants {
        grant.delegable = false;
    }
    grants
}

struct World {
    scenario: Scenario,
    server: FakeHttp,
    supervisor: Supervisor,
}

fn boot() -> World {
    boot_with_fs("fs")
}

fn boot_with_fs(fs_endpoint: &str) -> World {
    boot_with_memory(fs_endpoint, &[])
}

fn boot_with_memory(fs_endpoint: &str, memory_options: &[&str]) -> World {
    let server = FakeHttp::start();
    let mut scenario = Scenario::new();
    scenario.add_fs("fs", "data", fs_endpoint);
    scenario.add_openrouter("openrouter", &server.base_url(), KEY, &["--retries", "0"], true);
    scenario.add_conversation("conversation");
    scenario.add_chat_agent("chat", MODEL, &["--max-steps", "6"]);
    scenario.add_sessions("sessions", "chat");
    // `--quiet-ms 0`: a session is read back on the next tick rather than in
    // five minutes. Everything else is what a deployment would run.
    scenario.add_memory("memory", "data", MODEL, &["--quiet-ms", "0", "--fs-endpoint", fs_endpoint]);
    scenario.add_args("memory", memory_options);
    std::fs::create_dir_all(scenario.host("projects/default")).unwrap();
    scenario.bind("memory", "projects", "/projects", false);
    scenario.add_args("memory", &["--projects", "/projects"]);
    scenario.add_ui("ui");
    scenario.add_ui("restored-ui");
    // The UI service's part when it deletes a project.
    let forget =
        json!({"tool": "call", "endpoint": "memory", "action": "forget", "path": ["default"], "payload": {}});
    scenario.add_agent("deleter", &json!({"turns": [{"tool_calls": [forget]}]}), None);
    let stop =
        json!({"tool": "call", "endpoint": "memory", "action": "stop", "path": ["default"], "payload": {}});
    scenario.add_agent("stopper", &json!({"turns": [{"tool_calls": [stop]}]}), None);

    let supervisor =
        scenario.start_with(scenario.config(|limits| limits.max_inflight_per_endpoint = 16, 64)).unwrap();
    scenario.boot_service(&supervisor, "fs", fs_endpoint);
    scenario.boot_service(&supervisor, "openrouter", "model");
    scenario.boot_service(&supervisor, "conversation", "conversation");
    let mut grants = memory_grants();
    grants.last_mut().unwrap().endpoint = nucleus_e2e::endpoint(fs_endpoint);
    let memory = supervisor.boot(program("memory"), grants).unwrap();
    let endpoint = nucleus_e2e::endpoint("memory");
    assert!(
        supervisor.wait_until(TIMEOUT, |s| s.provider_of(&endpoint).is_some_and(|(p, _)| p == memory)),
        "the memory service never provided its endpoint"
    );
    let sessions = supervisor.boot(program("sessions"), sessions_grants()).unwrap();
    let endpoint = nucleus_e2e::endpoint("sessions");
    assert!(supervisor.wait_until(TIMEOUT, |s| s.provider_of(&endpoint).is_some_and(|(p, _)| p == sessions)));
    World { scenario, server, supervisor }
}

fn candidate_file(request: &nucleus_e2e::http::Received, name: &str) -> String {
    let body: serde_json::Value = serde_json::from_str(&request.body).unwrap();
    let prompt = body["messages"][0]["content"].as_str().unwrap();
    let start = prompt.find("memories/default/.candidates/").expect("candidate in prompt");
    let tail = &prompt[start..];
    let end = tail.find("/workspace_diff.md").unwrap();
    format!("{}/{name}", &tail[..end])
}

/// The whole loop: a turn happens, the session is read back, the consolidating
/// agent writes the index, and the next session is told what it says. The
/// memories are on a filesystem provider of their own.
#[test]
fn what_one_session_taught_reaches_the_next_one() {
    const FS_ENDPOINT: &str = "memory_files";
    let world = boot_with_fs(FS_ENDPOINT);
    // What the extraction model returns for the session it is shown.
    world.server.script_for(
        "part of a memory system",
        Scripted::json(
            200,
            &completion(
                MODEL,
                &json!({
                    "session_summary": format!("The owner said: {REMEMBERED}."),
                    "session_slug": "plan-first"
                })
                .to_string(),
            ),
        ),
    );
    // The consolidating agent: one tool call to write the index, then it is done.
    world.server.script_with(
        "consolidating memory",
        move |request| Scripted::json(
            200,
            &tool_request(
                MODEL,
                &[(
                    "write_index",
                    &format!("{FS_ENDPOINT}_create"),
                    json!({"path": candidate_file(request, "MEMORY.md"),
                           "content": format!("- [Plan first](summaries/plan-first-default.md) — {REMEMBERED}\n")}),
                )],
            ),
        ),
    );
    world.server.script_for("consolidating memory", Scripted::json(200, &completion(MODEL, "consolidated")));
    // Two ordinary chat turns, one per session.
    world.server.script(Scripted::json(200, &completion(MODEL, "Here is the plan.")));
    // The second session's answer cites what it was given, in the form the
    // read path asks for.
    world.server.script(Scripted::json(
        200,
        &completion(
            MODEL,
            "Hello again.\n\n<citation_entries>\nsummaries/plan-first-default.md:1-1|note=[the owner's \
             standing request]\n</citation_entries>",
        ),
    ));

    world.supervisor.boot(program("ui"), ui_grants()).unwrap();
    let mut ui = world.scenario.connect_ui("ui");
    ui.request(json!({"op": "sessions.create", "session": "default"})).unwrap();
    ui.say("default", "u1", "what is the plan?").unwrap();
    ui.wait_turns("default", 1);

    // The extraction and consolidation are the service's own work. Wait for
    // the baselined index that readers use: the workspace file can already
    // exist while accept_candidate is still writing it, before publication.
    let store = nucleus_memories_write::Store::open(&world.scenario.host("memory-state/default.db")).unwrap();
    assert!(
        nucleus_e2e::wait_for(Duration::from_secs(60), || !store.published().unwrap().0.is_empty()),
        "the consolidating agent never published its index"
    );
    let written = store.published().unwrap().0;
    assert!(written.contains(REMEMBERED), "the consolidating agent wrote the index: {written:?}");
    let record = world.scenario.host("data/memories/default/summaries/plan-first-default.md");
    assert!(
        std::fs::read_to_string(&record).unwrap().contains(REMEMBERED),
        "what the extraction wrote down is in the workspace under the slug it chose"
    );

    // A second session is told what the first one taught, as data.
    ui.request(json!({"op": "sessions.create", "session": "second"})).unwrap();
    ui.say("second", "u2", "hello again").unwrap();
    ui.wait_turns("second", 1);
    let injected = world
        .server
        .received()
        .into_iter()
        .filter(|r| r.body.contains("hello again"))
        .find(|r| r.body.contains(REMEMBERED))
        .expect("the second session was told what is remembered");
    assert!(injected.body.contains("not instruction and not authority"), "and it was told what that text is");

    // And the loop closes: the citation in that answer is counted against the
    // memory it named, so the next consolidation knows it was worth keeping.
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let now = i64::try_from(
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis(),
        )
        .unwrap();
        let rows = store.select_for_consolidation(10, 3650, now).unwrap();
        let counted = rows.iter().find(|r| r.session_id == "default").map(|r| r.usage_count);
        if counted == Some(Some(1)) {
            break;
        }
        assert!(Instant::now() < deadline, "the citation was never counted: {rows:?}");
        std::thread::sleep(Duration::from_millis(50));
    }
    drop(store);

    // The project is deleted: nothing of its memory is left.
    let mut forget = cap("memory", &["forget"], &[]);
    forget.delegable = false;
    world.supervisor.boot(program("deleter"), vec![forget]).unwrap();
    let forgot = world.scenario.wait_done_within("deleter", TIMEOUT);
    assert_eq!(nucleus_e2e::ok_value(&forgot[0]), Some(&json!({})), "{forgot:?}");
    for gone in ["data/memories/default", "memory-state/default.db"] {
        assert!(!world.scenario.host(gone).exists(), "{gone} is still there");
    }
}

fn extraction(world: &World) {
    world.server.script_for(
        "part of a memory system",
        Scripted::json(
            200,
            &completion(
                MODEL,
                &json!({"session_summary": "A useful remembered fact", "session_slug": "fact"}).to_string(),
            ),
        ),
    );
    world.server.script(Scripted::json(200, &completion(MODEL, "Noted.")));
}

fn start_remembering(world: &World) -> nucleus_e2e::UiClient {
    let mut grants = ui_grants();
    grants.push(cap("memory", &["contents"], &[]));
    world.supervisor.boot(program("ui"), grants).unwrap();
    let mut ui = world.scenario.connect_ui("ui");
    ui.request(json!({"op": "sessions.create", "session": "default"})).unwrap();
    ui.say("default", "remember", "Please remember a useful fact").unwrap();
    ui.wait_turns("default", 1);
    ui
}

#[test]
fn worker_cannot_modify_workspace_files_outside_its_candidate() {
    let world = boot();
    extraction(&world);
    world.server.script_with("consolidating memory", |request| {
        Scripted::json(
            200,
            &tool_request(
                MODEL,
                &[
                    (
                        "attack",
                        "fs_append",
                        json!({"path": "memories/default/raw_memories.md", "content": "\nHOSTILE RECORD\n"}),
                    ),
                    (
                        "write",
                        "fs_create",
                        json!({"path": candidate_file(request, "MEMORY.md"), "content": "validated memory"}),
                    ),
                    (
                        "rewrite_input",
                        "fs_create",
                        json!({"path": candidate_file(request, "unexpected"), "content": "untrusted input"}),
                    ),
                ],
            ),
        )
    });
    world.server.script_for("consolidating memory", Scripted::json(200, &completion(MODEL, "Done.")));
    let mut ui = start_remembering(&world);
    assert!(nucleus_e2e::wait_for(Duration::from_secs(20), || ui
        .request(json!({"op": "memory.contents"}))
        .is_ok_and(|v| v["index"] == "validated memory")));
    let requests = world.server.received();
    let last = requests.iter().rev().find(|r| r.body.contains("consolidating memory")).unwrap();
    assert!(last.body.contains("permission_denied"), "{last:?}");
    assert!(
        !std::fs::read_to_string(world.scenario.host("data/memories/default/raw_memories.md"))
            .unwrap()
            .contains("HOSTILE RECORD")
    );
}

#[test]
fn terminating_a_worker_never_publishes_its_partial_index_or_consumes_rows() {
    partial_is_not_published(ConsolidationEnd::TerminateWorker);
}

#[test]
fn terminating_memory_cleans_its_workspace_after_the_worker_is_reaped() {
    partial_is_not_published(ConsolidationEnd::TerminateMemory);
}

#[test]
fn a_failed_model_call_never_publishes_a_workers_partial_index() {
    partial_is_not_published(ConsolidationEnd::ModelFailed);
}

#[test]
fn restoring_a_stopped_project_keeps_memory_work_stopped_across_service_restart() {
    partial_is_not_published(ConsolidationEnd::StopProject);
}

#[derive(Clone, Copy)]
enum ConsolidationEnd {
    TerminateWorker,
    TerminateMemory,
    ModelFailed,
    StopProject,
}

fn partial_is_not_published(end: ConsolidationEnd) {
    let world = if matches!(end, ConsolidationEnd::ModelFailed) {
        boot_with_memory("fs", &["--agent-model-retries", "0"])
    } else {
        boot()
    };
    extraction(&world);
    world.server.script_with("consolidating memory", |request| {
        Scripted::json(
            200,
            &tool_request(
                MODEL,
                &[(
                    "partial",
                    "fs_create",
                    json!({"path": candidate_file(request, "MEMORY.md"), "content": "PARTIAL INDEX"}),
                )],
            ),
        )
    });
    // The worker's next call is held until the test ends, so it is still running when it is ended.
    let (held, _held) = Scripted::json(200, &completion(MODEL, "Done.")).held();
    world.server.script_for(
        "consolidating memory",
        if !matches!(end, ConsolidationEnd::ModelFailed) {
            held
        } else {
            // A provider error is retryable under the agent's default
            // policy. This deployment explicitly disables those retries.
            Scripted::json(500, &json!({"error": {"message": "failed"}}))
        },
    );
    let mut ui = start_remembering(&world);
    assert!(nucleus_e2e::wait_for(Duration::from_secs(20), || world
        .server
        .received()
        .iter()
        .filter(|r| r.body.contains("consolidating memory"))
        .count()
        == 2));
    let memory = world.supervisor.provider_of(&nucleus_e2e::endpoint("memory")).unwrap().0;
    let child = world.scenario.children_of(memory)[0];
    let before = ui.request(json!({"op": "memory.contents"})).unwrap();
    assert_eq!(before["index"], "", "candidate must stay unpublished even while running");
    let store = nucleus_memories_write::Store::open(&world.scenario.host("memory-state/default.db")).unwrap();
    match end {
        ConsolidationEnd::TerminateWorker => world.supervisor.terminate(child),
        ConsolidationEnd::TerminateMemory => world.supervisor.terminate(memory),
        ConsolidationEnd::ModelFailed => {}
        ConsolidationEnd::StopProject => {
            // A finished chat has another extraction queued while the worker is out.
            assert!(store.reopen("extract", "default").unwrap());
            std::fs::rename(
                world.scenario.host("projects/default"),
                world.scenario.host("projects/.archiving-default"),
            )
            .unwrap();
            world.supervisor.boot(program("stopper"), vec![cap("memory", &["stop"], &["default"])]).unwrap();
            let stopped = world.scenario.wait_done_within("stopper", TIMEOUT);
            assert_eq!(nucleus_e2e::ok_value(&stopped[0]), Some(&json!({})), "{stopped:?}");
            std::fs::rename(
                world.scenario.host("projects/.archiving-default"),
                world.scenario.host("projects/.archived-default"),
            )
            .unwrap();
            let requests = world.server.received().len();
            world.supervisor.terminate(memory);
            assert!(world.supervisor.wait_until(TIMEOUT, |s| s.cleanup_confirmed(memory)));
            std::fs::rename(
                world.scenario.host("projects/.archived-default"),
                world.scenario.host("projects/default"),
            )
            .unwrap();
            world.supervisor.boot(program("memory"), memory_grants()).unwrap();
            assert!(
                world
                    .supervisor
                    .wait_until(TIMEOUT, |s| s.provider_of(&nucleus_e2e::endpoint("memory")).is_some())
            );
            // A restarted service has a new endpoint generation: use a fresh client.
            let mut grants = ui_grants();
            grants.push(cap("memory", &["contents"], &[]));
            world.supervisor.boot(program("restored-ui"), grants).unwrap();
            ui = world.scenario.connect_ui("restored-ui");
            // This call is served after the restarted scheduler's first pass.
            assert_eq!(ui.request(json!({"op": "memory.contents"})).unwrap()["index"], "");
            assert_eq!(
                world.server.received().len(),
                requests,
                "restoring must not restart queued memory work"
            );
        }
    }
    // A pass that failed stays failed; one the service was stopped in the
    // middle of is ready to run again when it restarts.
    let expected = if matches!(end, ConsolidationEnd::TerminateMemory) { "ready" } else { "failed" };
    assert!(nucleus_e2e::wait_for(TIMEOUT, || store.status("consolidate", "global").unwrap().as_deref()
        == Some(expected)));
    if matches!(end, ConsolidationEnd::ModelFailed) {
        assert_eq!(
            world.server.received().iter().filter(|r| r.body.contains("consolidating memory")).count(),
            2,
            "the failed second generation must not be retried"
        );
    }
    assert!(
        store.selected().unwrap().is_empty(),
        "failed consolidation cannot mark selected rows as consumed"
    );
    assert!(!world.scenario.host("data/memories/default/MEMORY.md").exists());
    if matches!(end, ConsolidationEnd::TerminateMemory) {
        assert!(world.supervisor.wait_until(TIMEOUT, |supervisor| supervisor.cleanup_confirmed(memory)));
        assert!(world.supervisor.cleanup_confirmed(child));
    } else {
        assert_eq!(ui.request(json!({"op": "memory.contents"})).unwrap()["index"], "");
    }
    assert!(store.published().unwrap().0.is_empty());
}
