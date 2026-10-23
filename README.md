# Open Math Harness

## This fork: local macOS + Codex research

This fork adds a standalone **paper-math composition** using Python 3.10+ and an
installed, logged-in Codex CLI. It runs on macOS without Rust, bubblewrap or an Open
Math OS account. It reuses the upstream `pi0` and `pi_doubling` discovery policies
and saves a search tree, input snapshots, candidate reports and execution events.

```sh
codex login
python3 tools/codex_harness.py doctor
python3 tools/codex_harness.py run \
  --task-file /path/to/task.txt \
  --context-file /path/to/known-results.txt \
  --run-dir .codex-harness/experiment-1
python3 tools/codex_harness.py resume \
  --run-dir .codex-harness/experiment-1 \
  --followup 'Check the previous argument and resolve its first open obligation.'
python3 tools/codex_harness.py status .codex-harness/experiment-1
```

The same commands can be run by a local Codex app task. Context files are optional
and repeatable. `--rounds`, `--width` and `--timeout` bound execution; width defaults
to one and attempts run sequentially. `--policy pi_doubling` selects the upstream
fresh-root/continuation mix. `--model` is optional; omission uses the CLI default.
Normal Codex account usage applies. No credentials are embedded in this fork.

Workers receive explicit text snapshots. They use the Codex read-only sandbox,
with shell, apps, plugins, hooks, browser tools and delegation disabled. This mode
does not run Lean or edit the source project. User configuration is not loaded;
saved CLI authentication is reused. Resume verifies snapshot and candidate hashes
and never silently replaces the original target. Results are **unverified**, even
if the model reports a complete proof. Timeouts, errors and interruptions remain
recoverable failed attempts; they cannot be reported as solved proofs.

This is a new local composition, **not a macOS port of the Rust microkernel** or
the hosted UI, memory service, multi-tenant isolation or Lean verifier. The Linux
runtime below retains its original platform requirements. Codex's sandbox is a
different trust boundary from Nucleus capabilities and Linux namespaces. The
module documentation in `tools/codex_harness.py` describes the exact limitations.
Run directories contain private research and logs; keep them outside publication.

Portable regression tests (Python 3.12+, no model call):

```sh
python3 -m unittest discover -s tools -p 'test_*.py'
python3 tools/check_markdown.py
```

Codex integration references: [non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode)
and [integration choices](https://developers.openai.com/blog/codex-as-a-platform).

## Upstream runtime

### Local Linux development on macOS

With Lima and Python 3.12+ installed, this fork can build and test the Linux
runtime in a dedicated VM on the Mac. Commit intended source changes first:
`sync` transfers a committed tree, excluding uncommitted and ignored files.

```sh
python3 tools/local_runtime.py start
python3 tools/local_runtime.py sync
python3 tools/local_runtime.py check --suite core
python3 tools/local_runtime.py slice
python3 tools/local_runtime.py check --suite all
python3 tools/local_runtime.py stop
```

The VM uses Lima plain mode, with no guest-agent vsock tunnel, host directory
mounts or forwarded SSH agent. An older development VM can be migrated with
`python3 tools/local_runtime.py harden`; its disk and data are retained. Runtime processes
retain Linux bubblewrap confinement. Stopping keeps its disk and source snapshots.
The operator workflow and failure behavior are documented in
`tools/local_runtime.py` and `deploy/local/guest.py`. The local resident
composition below is under integration testing; it is not yet a verified
replacement for the hosted composition.

An explicit Codex integration check is also available after `sync` and `build`:

```sh
python3 tools/local_runtime.py model-smoke --codex /path/to/codex
```

This consumes the logged-in host account's usage. It starts a temporary confined
Linux chat composition, checks an allowed read and a kernel denial, restarts the
runtime and verifies that saved history continues without replaying the read.
Host CLI credentials stay on the Mac. `tools/local_model.py` documents the local
broker's completion-only text protocol, parameter limits and cancellation.
This small smoke test does not exercise project identities, memory or Lean.

`python3 tools/local_runtime.py prepare-lean` installs the pinned public Lean,
Mathlib and REPL development environment and publishes a separate root-owned,
read-only copy for the verifier. The runtime must be stopped. It requires at least
30 GiB of free guest disk space for a new toolchain; new VMs have 12 GiB of memory
and a 64 GiB disk. For an older smaller VM, stop it and run
`limactl edit open-math-harness --memory 12` before starting it again: the resident
REPL and independent verifier need room to coexist. The Mac must also have
memory available for macOS and other applications. Guest free memory and zero
guest swap do not rule out host swapping, which can stall Mathlib imports past
their deadlines. Check the Mac's memory pressure before increasing the VM's RAM.
If builds leave the host under pressure, stop the dedicated VM before freeing
host memory, then start it fresh; this ends every guest process and keeps its disk.
The opt-in tests use the development assets; publication alone does not configure
or start a resident verifier.
After installation, `python3 tools/local_runtime.py check --suite lean` runs the
existing opt-in verifier, prover, discovery and plan coordination tests with real
Lean and deterministic model responses. Missing dependencies fail the command.
`python3 tools/local_runtime.py prepare-platform` builds and installs the runtime
programs, prepares the real runtime/projects Unix identities, and runs the six
opt-in identity and cgroup tests under systemd delegation. It retains existing
workspace data and does not start a resident composition. After both preparation
commands, the persistent composition can be run in one terminal:

```sh
python3 tools/local_runtime.py serve --codex /path/to/codex
```

Wait for the `ready` event, then use another terminal:

```sh
python3 tools/local_runtime.py chat --project default --session default
```

`--message 'text'` sends one message. `request` reads one JSON UI operation from
stdin; the operation names and payloads are documented in
`program-files/ui/src/lib.rs`. Both clients use the owner-only guest Unix socket;
requests still go through the UI service's kernel handles. Leaving the chat
client does not stop its session. Ctrl-C in `serve` stops the composition and its
bridge; its persistent conversations, project files and memory remain in the VM.
The transient systemd unit also depends on that dedicated SSH login's lifetime.
Startup and cleanup failures are reported. The bridge does not reconnect or
replay requests; recorded memory jobs retain the memory service's restart
behavior. Inspect the guest's `journalctl -u nucleus-local` on failure.

The composition includes the ordinary project, conversation, memory and Lean
discovery services. Their live local workflows are still being verified.
The host CLI proposes tool calls as structured text; Nucleus authorizes and
executes them in Linux. Only `codex/default` is exposed; `serve --model` selects
the actual host Codex model. Claude is not implemented yet. Images, explicit
output-token caps, temperature and opaque reasoning state are unsupported.
Reported token usage is retained, but dollar cost is unknown: discovery has no
USD ceiling and uses its existing per-task model-call budgets. Hosted platform
and GitHub links, suggestions and Ultramath are omitted; the latter two require
output-token caps that this CLI bridge cannot enforce. Project archiving and
unarchiving are unavailable because the UI's archive workflow requires swarm.

**Open Math Harness** is the open-source runtime of
[Open Math OS](https://www.openmathmodel.org/), and this repository holds its kernel,
services and agents, written in Rust. Its first generation is named Nucleus.

Open Math OS is infrastructure for Math 2.0: the operating system for collaborative math
research, with open frontier models, a scientific compute cloud and collaboration tools
in one system, and the mathematical community in control of its data.

On Open Math OS, Open Math Harness runs agents on open models, parallel
exploration with Ultramath and proofs checked by Lean. Its capability-based microkernel
holds every agent to the authority it was given: agents propose, mathematicians decide.

[Apply for a team on Open Math OS](https://apply.openmathmodel.org/teams).

## Project status

Open Math Harness is developed as the runtime of Open Math OS. We publish its source so
that its design and implementation can be inspected, studied and built on.

> [!WARNING]
> **The full runtime currently relies on the Open Math OS deployment environment.**
> This repository supports building the runtime and running a small self-contained
> example, but does not yet provide a standalone local setup for interactive agents
> or proof search.

We plan to reduce these deployment dependencies and make individual components easier
to run independently in future releases.

## What it runs

- **Agents on open models.** Every chat has an agent on the open-weight model and the
  reasoning depth chosen for the task, with streaming replies, tool use, sub-agents and
  persistent session history.
- **Ultramath.** Several agents explore different approaches to one hard problem and
  bring their findings back to the chat. You decide what holds.
- **Proof search.** A tree search over candidate proofs of a fixed Lean statement, with
  confined prover agents working its branches in parallel.
- **Proofs checked by Lean.** Every candidate is checked by Lean in a confined workspace,
  with a deadline. Whether a proof holds is decided by Lean, never by what a model says
  about it.
- **Tools and memory for every project.** Projects are git repositories. Agents work with
  files and the shell, web and PDF tools, the project's skills, MCP servers (stdio and
  Streamable HTTP) and plugins they build, and keep a project memory across sessions.
- **People in charge.** A protected action waits for a person's approval, and a session's
  records are scanned for secrets before the platform shows them.

## Built for everyday math research

Agents propose; mathematicians review and decide. Open Math Harness is designed so that
this holds in the runtime, not only in the prompts:

- Every program, agents included, is a confined process that holds only the capabilities
  its composition grants it, and what it starts receives no more.
- Model output, tool results and repository content are untrusted input. Text cannot grant
  authority, approve an action or decide a verification result.
- Credentials stay with narrowly privileged providers. They never reach a model, a
  transcript or a log.
- Agent loops, model providers, tools and memory are replaceable services. The kernel keeps
  only the mechanisms every service must share: identity, capabilities, IPC and lifetimes.

`AGENTS.md` states these requirements; each module's docs say what its code guarantees.

## Design

### Ultramath

Ultramath coordinates multiple agents exploring different approaches to a hard problem.
Each member registers a proposed direction, which a model-based duplicate check compares
against other registered directions. Members share lemmas, partial results,
counterexamples and unsuccessful approaches on a common board, and keep their working
files in separate directories.

Exploration runs asynchronously alongside the conversation. Members submit reports and
supporting files to the coordinating agent for review. The coordinator can return
feedback for revision or accept a solution, which stops the remaining members. The run
records directions, findings and review decisions for later inspection.

### Proof search

Proof search starts from a fixed Lean statement and preserves candidate proofs in a
tree. An exploration policy selects which saved workspaces to continue, and workers
explore those branches in parallel. The harness evaluates submitted candidates through
the Lean verifier and records the results, so later rounds can build on earlier work.

The `discovery` interface manages runs and exposes their progress and search trees.
The `explore` interface gives each worker its assignment, lets workers share research
plans, and receives their candidates. A proof counts as solved only when it passes
verification against the original statement.

### Memory

Memory is maintained per project. After a session becomes quiet, the memory service
extracts material from its history and consolidates selected material into `MEMORY.md`.
Later sessions receive that memory as context and cite what they use; citation history
helps select material for future consolidation. Memory carries no authority to grant
permissions or override the user's current instructions.

## Building from source

### Requirements

- Linux, with a C compiler and linker for native dependencies. Windows and macOS are
  not supported as native runtime hosts.
- Rust installed through rustup. [rust-toolchain.toml](rust-toolchain.toml) pins the
  toolchain and includes the formatting and linting components.
- [bubblewrap](https://github.com/containers/bubblewrap), with `bwrap` on `PATH` and
  permission to create unprivileged user namespaces. Host security policy must allow
  bubblewrap to create its sandbox; startup fails if confinement is unavailable.

### Build

```sh
git clone https://github.com/openmathmodel/open-math-harness.git
cd open-math-harness
cargo build --locked --workspace
```

This builds the launcher and service binaries into `target/debug/`.

Running interactive agents additionally requires a service configuration, model provider
credentials and a client. Proof search also requires a configured Lean and Mathlib
environment. The files in `deploy/vm/` target the managed Open Math OS VM and are not
a standalone installer for a personal machine.

### Run the local example

The [slice example](examples/slice/) runs a scripted agent, a worker and a filesystem
service to demonstrate capability checks and delegation. It does not call a model or
start interactive chat, Ultramath, memory or proof search. It needs no model credentials
or Open Math OS deployment, but still requires Linux and working bubblewrap confinement.
After building, run this command from the repository root:

```sh
cargo run --locked --manifest-path cli/Cargo.toml -- run examples/slice/config.json
```

The agent reads permitted files, delegates access to a smaller directory to its worker,
and attempts operations that the kernel refuses. Calls and results are recorded in
`examples/slice/out/agent/transcript.jsonl` and
`examples/slice/out/agent/worker/transcript.jsonl`. The filesystem service records the
calls it receives in `examples/slice/log/access.jsonl`.

The example also creates `examples/slice/data/work/report.txt`. To repeat that creation
successfully, remove the generated report before running the example again.

## Development

Run the Rust tests from the repository root after building:

```sh
cargo test --workspace
```

The default tests require no paid model API or production credentials. Integration
tests exercise real child processes and bubblewrap; tests requiring external providers
or a Lean installation are explicitly opt-in.

The [CI workflow](.github/workflows/ci.yml) defines the full checks and their environment,
including formatting, Clippy, Python tooling tests and Node.js MCP integration tests.
See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution requirements, including AI
review, testing and the information to include in a pull request.

## Community

We welcome discussion and contributions. Join the
[Open-Math-Model channel](https://zulip.sair.foundation/#narrow/channel/27-Open-Math-Model)
on the SAIR Foundation Zulip, and see
[Community discussion](CONTRIBUTING.md#community-discussion) for how to take part.

## License

Open Math Harness is licensed under the [Apache License 2.0](LICENSE).
Attributions and license terms for incorporated third-party material are in
[third-party/NOTICES.txt](third-party/NOTICES.txt).

## Version

The current release is Open Math Harness 0.41.0, the Nucleus generation.
