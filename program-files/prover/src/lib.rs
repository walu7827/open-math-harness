//! The prover agent: the discovery agent for theorem proving.
//!
//! One process, one continuation of one problem in the Dream-RSI discovery
//! tree. It asks the model for a complete candidate `.lean` file, has the
//! confined `lean` service judge it, and — if rejected — feeds the verdict and
//! Lean's diagnostics back and asks again. It **never stops on its own**:
//! unlike a chat turn, which ends when the model stops calling tools, an
//! attempt ends only on a passing verdict, on the attempt or model budget, on
//! a model error, when no prompt can be made from the template, or on a
//! terminate request: persistence is the loop's structure, not a nudge.
//!
//! The task is selected **by authority, not by a flag**: the process holds an
//! `explore submit` capability whose resource is `[key, problem, task]`, `key`
//! the run's (see
//! `nucleus-explore-interface`); one without it does not start. The worker
//! asks `explore assignment` for the parent node's workspace (its file, the
//! checker's verdict on it, what the siblings already tried, and the run's
//! prompt template), then runs the persistent model→check loop — the evaluator
//! is `lean check`, which allows `sorry` and lists the open goals — for at most
//! `min(--max-attempts, assignment.model_calls)` rounds, keeps the best
//! candidate (`Complete > Sorries > Errors > Invalid`, ties to the latest),
//! stops early on `Complete`, and ends in **at most one** `explore submit` —
//! called by the model itself when it hands its file over (the tool loop), and
//! by the worker on the model's behalf when the budget ran out first. The
//! harness, not the worker, scores the node: what is submitted is a candidate,
//! never a verdict.
//!
//! Authority of this fixed loop: the model capability (narrowed to one model,
//! budgeted by the kernel's `max_calls`), `lean check` and the task's
//! `explore` capabilities. Nothing else — no filesystem, no shell, no network.
//! A `shell exec` capability scoped to a workspace selects the tool loop
//! ([`agent`]) instead. The model decides what to try; it cannot verify
//! anything itself, and a "solved" claim in its text counts for nothing.
//!
//! # The system prompt is a template
//!
//! The task definition — the rules the model works under, the statement, the
//! parent's file, the checker's verdict and what the siblings tried — is a
//! Jinja template rendered into the **system message** by [`PromptTemplate`]:
//! `minijinja`, strict (an undefined variable is an error), no autoescape, no
//! includes. The variables are listed on [`PromptVars`]. The first user
//! message is the fixed [`FIRST_USER_MESSAGE`]; the (candidate, verdict)
//! exchanges follow as assistant/user turns. The templates ship in
//! `program-files/prover/prompts/`: `putnam.j2` for this loop,
//! `prover-agent.j2` for the tool loop; the worker gets its template from the
//! assignment. There is no built-in prompt: an empty template, one that does
//! not parse, or one that does not render fails the attempt before any model
//! call, with [`PROMPT_TEMPLATE_FAILURE`] as the recorded reason, and ends the
//! run. A render is bounded ([`PROMPT_FUEL`] instructions,
//! [`MAX_SYSTEM_MESSAGE_BYTES`] of output): a template that loops without end
//! or prints without end fails the same way instead of pinning the worker.
//!
//! Output (durable, in `--out`): `attempts.jsonl`, one line per candidate with
//! the verdict, and `result.json` at the end, under `out/<problem>/<task>/`.
//! The result identifies its operation with `mode: "continue"` in both loops.
//! This record is the worker's report, not verification authority; the run
//! still scores the submitted candidate through its independent Lean handle.
//!
//! Model output and Lean diagnostics are untrusted text; the template is
//! trusted input from the task's author, the values rendered into it are not.
//! `--max-attempts` and the model handle's call budget bound model calls. Output
//! budgets are optional; the provider supplies its default when none is chosen.
//! The fixed loop keeps only the last few exchanges of a long run.

pub mod agent;
mod agent_prompts;
mod model_call;
mod planning;
pub mod transcript;

pub(crate) use model_call::complete;
pub use model_call::{ModelAccess, Progress};

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use minijinja::{AutoEscape, Environment, UndefinedBehavior};
use nucleus_explore_interface as explore;
use nucleus_lean_interface as lean;
use nucleus_model_interface::{self as model, FinishReason, Message, ReasoningEffort, Role};
use nucleus_protocol::{Error, ErrorCode, HandleId, HandleInfo, ResourcePath};
use nucleus_sdk::{ChannelError, Connection};
use serde::Serialize;

#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub model: String,
    /// Results go to `out/<problem>/<task>/`, so many workers may share one `out`.
    pub out: PathBuf,
    pub max_attempts: u32,
    pub max_output_tokens: u32,
    /// Sampling temperature; absent leaves the provider default.
    pub temperature: Option<f32>,
    pub model_deadline_ms: u64,
    /// Additional attempts for a failed model step in the tool loop. Each
    /// attempt consumes the worker's existing model-call budget.
    pub model_retries: u32,
    /// Wait between those attempts; termination interrupts the wait.
    pub model_retry_backoff_ms: u64,
    /// Deadline for one `lean verify` or `lean check` call.
    pub verify_deadline_ms: u64,
    pub lean_endpoint: String,
    /// The endpoint of the `explore` interface, whose `submit` capability
    /// names this worker's task.
    pub explore_endpoint: String,
    /// Reasoning effort asked of the model; `None` leaves the provider default.
    /// The lever for cost and latency on a reasoning model.
    pub reasoning_effort: Option<ReasoningEffort>,
    /// The endpoint of the `shell` interface; holding an `exec` capability on
    /// it is what selects the tool loop ([`agent`]).
    pub shell_endpoint: String,
    /// Wall clock a model's `shell exec` gets when it names no `timeout_ms`
    /// (the tool loop), which the shell clamps to its maximum.
    pub bash_timeout_ms: u64,
    /// The context size at which the tool loop resets its context.
    pub compact_at_tokens: u64,
    /// How long a streamed call may produce nothing at all before this worker
    /// cancels it and reports a timeout. Thinking counts as producing, so
    /// this is silence rather than quiet ([`model_call::STALL_MS`] unless given).
    pub stall_ms: u64,
    /// The project this worker's model calls are made for (`--project`, a
    /// discovery run's). Metadata for the provider's records, never authority.
    pub project: Option<String>,
}

impl Options {
    /// Parses `--model <id> --out <dir>
    /// [--max-attempts <n>] [--max-output-tokens <n>] [--model-deadline-ms <n>]
    /// [--verify-deadline-ms <n>] [--lean-endpoint <name>]
    /// [--explore-endpoint <name>] [--reasoning-effort off|on|low|medium|high|max]
    /// [--temperature <0..2>] [--shell-endpoint <name>] [--bash-timeout-ms <n>]
    /// [--compact-at-tokens <n>] [--model-retries <n>] [--model-retry-backoff-ms <n>]
    /// [--stall-ms <n>] [--project <name>]`.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self, String> {
        let (mut model, mut out) = (None, None);
        let mut options = Options {
            model: String::new(),
            out: PathBuf::new(),
            max_attempts: 8,
            max_output_tokens: 0,
            temperature: None,
            model_deadline_ms: 300_000,
            model_retries: 3,
            model_retry_backoff_ms: 2_000,
            verify_deadline_ms: 300_000,
            lean_endpoint: lean::ENDPOINT.to_owned(),
            explore_endpoint: explore::ENDPOINT.to_owned(),
            reasoning_effort: None,
            shell_endpoint: nucleus_shell_interface::ENDPOINT.to_owned(),
            bash_timeout_ms: agent::DEFAULT_BASH_TIMEOUT_MS,
            compact_at_tokens: agent::DEFAULT_COMPACT_AT_TOKENS,
            stall_ms: model_call::STALL_MS,
            project: None,
        };
        let number = |flag: &str, value: &str| -> Result<u64, String> {
            value.parse().map_err(|_| format!("{flag} must be a number"))
        };
        let mut args = args.into_iter();
        while let Some(flag) = args.next() {
            let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
            match flag.as_str() {
                "--model" => model = Some(value),
                "--out" => out = Some(PathBuf::from(value)),
                "--max-attempts" => {
                    options.max_attempts = u32::try_from(number(&flag, &value)?).unwrap_or(u32::MAX)
                }
                "--max-output-tokens" => {
                    options.max_output_tokens = u32::try_from(number(&flag, &value)?).unwrap_or(u32::MAX)
                }
                "--model-deadline-ms" => options.model_deadline_ms = number(&flag, &value)?,
                "--temperature" => {
                    let temperature: f32 =
                        value.parse().map_err(|_| "--temperature must be a finite number between 0 and 2")?;
                    if !(0.0..=2.0).contains(&temperature) {
                        return Err("--temperature must be a finite number between 0 and 2".to_owned());
                    }
                    options.temperature = Some(temperature);
                }
                "--model-retries" => {
                    options.model_retries = u32::try_from(number(&flag, &value)?).unwrap_or(u32::MAX)
                }
                "--model-retry-backoff-ms" => options.model_retry_backoff_ms = number(&flag, &value)?,
                "--verify-deadline-ms" => options.verify_deadline_ms = number(&flag, &value)?,
                "--lean-endpoint" => options.lean_endpoint = value,
                "--explore-endpoint" => options.explore_endpoint = value,
                "--shell-endpoint" => options.shell_endpoint = value,
                "--bash-timeout-ms" => options.bash_timeout_ms = number(&flag, &value)?,
                "--compact-at-tokens" => options.compact_at_tokens = number(&flag, &value)?,
                "--stall-ms" => options.stall_ms = number(&flag, &value)?,
                "--project" => {
                    if !nucleus_protocol::valid_project(&value) {
                        return Err(format!("--project {value:?} is not a project's name"));
                    }
                    options.project = Some(value);
                }
                "--reasoning-effort" => {
                    options.reasoning_effort = Some(ReasoningEffort::parse(&value).ok_or_else(|| {
                        "--reasoning-effort must be off, on, low, medium, high or max".to_owned()
                    })?);
                }
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        options.model = model.ok_or("--model is required")?;
        options.out = out.ok_or("--out is required")?;
        if options.max_attempts == 0 {
            return Err("--max-attempts must be at least 1".to_owned());
        }
        if options.model_retry_backoff_ms > 60_000 {
            return Err("--model-retry-backoff-ms must be at most 60000".to_owned());
        }
        if options.bash_timeout_ms < 1_000 || options.compact_at_tokens < 1_000 {
            return Err("--bash-timeout-ms and --compact-at-tokens must be at least 1000".to_owned());
        }
        Ok(options)
    }
}

// ---------------------------------------------------------------------------
// The system-prompt template
// ---------------------------------------------------------------------------

/// The first user message of every prompt; all task-specific text is in the
/// rendered system message.
pub const FIRST_USER_MESSAGE: &str = "Return the complete file.";
/// Past (candidate, verdict) exchanges the fixed loop keeps in its prompt.
const KEEP_EXCHANGES: usize = 4;
/// The reason recorded on an attempt that never reached the model because the
/// template was empty, did not parse or did not render (`failure` in a
/// [`ContinueAttempt`]), and the `stopped` value of the run it ends.
pub const PROMPT_TEMPLATE_FAILURE: &str = "prompt_template";
/// Longest goal, error message, or sibling excerpt handed to the template.
const PROMPT_EXCERPT_CHARS: usize = 1500;
/// Most diagnostics of one kind, and most siblings, handed to the template.
const PROMPT_LIST_LIMIT: usize = 12;
/// Instructions one render may execute (`minijinja`'s fuel). The shipped
/// templates need a few thousand at most; a template that needs more is
/// looping, and fails the attempt.
pub const PROMPT_FUEL: u64 = 1_000_000;
/// Longest rendered system message, in bytes.
pub const MAX_SYSTEM_MESSAGE_BYTES: usize = 128 * 1024;

/// Why a prompt could not be produced. There is no fallback: each of these
/// fails the attempt and ends the run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptError {
    /// The template text is empty or whitespace (an assignment that carries
    /// none).
    Empty,
    /// The template does not parse.
    Syntax(String),
    /// Rendering failed: an undefined variable, a bad filter, a type error,
    /// the fuel spent, or the output over [`MAX_SYSTEM_MESSAGE_BYTES`].
    Render(String),
}

impl std::fmt::Display for PromptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PromptError::Empty => write!(f, "{PROMPT_TEMPLATE_FAILURE} is empty"),
            PromptError::Syntax(e) => write!(f, "{PROMPT_TEMPLATE_FAILURE} does not parse: {e}"),
            PromptError::Render(e) => write!(f, "{PROMPT_TEMPLATE_FAILURE} did not render: {e}"),
        }
    }
}

/// A parsed system-prompt template in its own environment: undefined
/// variables are errors, nothing is escaped, there are no other templates to
/// include and no loader, a render has [`PROMPT_FUEL`] instructions;
/// `trim_blocks` and `lstrip_blocks` are on so a line holding only a `{% tag
/// %}` leaves no blank line. The discovery service parses a task's template
/// in an environment set up the same way, so a syntax error is refused with
/// the task; only rendering can fail here.
pub struct PromptTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for PromptTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptTemplate").finish_non_exhaustive()
    }
}

impl PromptTemplate {
    const NAME: &'static str = "system";

    /// Parses `source`; empty or whitespace-only text is refused.
    pub fn new(source: &str) -> Result<Self, PromptError> {
        if source.trim().is_empty() {
            return Err(PromptError::Empty);
        }
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Strict);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_fuel(Some(PROMPT_FUEL));
        env.add_template_owned(Self::NAME.to_owned(), source.to_owned())
            .map_err(|e| PromptError::Syntax(format!("{e:#}")))?;
        Ok(Self { env })
    }

    /// The system message for one attempt. The output is written through a
    /// bounded sink, so a template that emits more than
    /// [`MAX_SYSTEM_MESSAGE_BYTES`] stops there rather than growing without
    /// limit.
    pub fn render(&self, vars: &PromptVars<'_>) -> Result<String, PromptError> {
        self.render_context(vars)
    }

    fn render_context(&self, vars: &impl Serialize) -> Result<String, PromptError> {
        let mut sink = BoundedSink::default();
        let rendered =
            self.env.get_template(Self::NAME).and_then(|t| t.render_captured_to(vars, &mut sink).map(drop));
        match (rendered, sink.overflow) {
            (_, true) => Err(PromptError::Render(format!(
                "the rendered system message exceeds {MAX_SYSTEM_MESSAGE_BYTES} bytes"
            ))),
            (Err(e), false) => Err(PromptError::Render(format!("{e:#}"))),
            (Ok(()), false) => {
                String::from_utf8(sink.bytes).map_err(|e| PromptError::Render(format!("not UTF-8: {e}")))
            }
        }
    }
}

/// Collects a render up to [`MAX_SYSTEM_MESSAGE_BYTES`] and refuses the rest.
#[derive(Default)]
struct BoundedSink {
    bytes: Vec<u8>,
    overflow: bool,
}

impl Write for BoundedSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if self.bytes.len().saturating_add(buf.len()) > MAX_SYSTEM_MESSAGE_BYTES {
            self.overflow = true;
            return Err(std::io::Error::other("system message too long"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// The template's variables, all always defined so a strict template can
/// test them. The bounded structures (`check`, `siblings`) are built once per
/// worker; only `calls_left` and `attempt` change between attempts, and
/// `putnam.j2` uses neither, so its system message is the same on every
/// attempt of a worker (the provider can cache the prefix).
#[derive(Clone, Debug, Serialize)]
pub struct PromptVars<'a> {
    /// `"continue"` in the fixed loop; in the tool loop, the
    /// message being rendered (`"system"`, `"task"`, `"resume"`, or a runtime
    /// mode).
    pub mode: &'static str,
    pub problem: &'a str,
    /// The frozen formalization.
    pub statement: &'a str,
    /// The parent's workspace.
    pub file: &'a str,
    /// The node being continued.
    pub parent: Option<u64>,
    /// Whether this worker starts from the parent's message transcript. When
    /// it does not, the `resume` message must not claim the workspace is "as
    /// you left it": the worker has never been here.
    pub inherited: bool,
    /// Lean's verdict on `file`, when it has one.
    pub check: Option<&'a CheckSummary>,
    /// The newest [`PROMPT_LIST_LIMIT`] continuations already made from the
    /// same parent.
    pub siblings: &'a [SiblingSummary],
    /// How many there are in all; the oldest beyond the bound are not listed.
    pub siblings_total: usize,
    /// Other completed branches of this run and problem (the assignment's,
    /// bounded by the explore interface): pointers for the `resume`
    /// message, never text from those branches.
    pub branch_references: &'a [explore::BranchReference],
    /// Model calls this worker may still make, counting the one being prepared.
    pub calls_left: u64,
    /// 1-based attempt within this worker.
    pub attempt: u32,
    /// The workspace file of the tool loop (`<problem>.lean`); the same name
    /// in every mode, so a template can mention it.
    pub file_name: &'a str,
    /// The tools the kernel lists for this worker (the tool loop), grouped by
    /// the service that provides them, with that service's own description
    /// of its environment. Empty for the fixed loop.
    pub services: &'a [ServiceSummary],
    /// The kernel-listed names of the plan tools, when this worker holds
    /// both; `None` otherwise, and always in the fixed loop.
    pub publish_plan_tool: Option<&'a str>,
    pub read_plans_tool: Option<&'a str>,
    /// The kernel-listed name of this worker's command tool (`exec` on
    /// `--shell-endpoint`, so not always `shell_exec`) and the limit a call
    /// of it without its own `timeout_ms` gets (`--bash-timeout-ms`), so the
    /// prompt can state both; `None` in the fixed loop.
    pub exec_tool: Option<&'a str>,
    pub bash_timeout_ms: Option<u64>,
}

/// One provider as the template sees it: its description and its tools.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ServiceSummary {
    pub endpoint: String,
    /// The provider's account of itself (`InterfaceDecl::description`).
    pub description: String,
    pub tools: Vec<ToolSummary>,
}

/// One kernel-listed tool as the template sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ToolSummary {
    /// `endpoint_action`, the name the model calls.
    pub name: String,
    pub description: String,
    /// The JSON Schema of the arguments, as text.
    pub params: String,
}

/// The kernel's listing for the template, grouped by endpoint in listing order.
pub fn summarize_services(tools: &[nucleus_protocol::ToolInfo]) -> Vec<ServiceSummary> {
    let mut services: Vec<ServiceSummary> = Vec::new();
    for t in tools {
        let summary = ToolSummary {
            name: nucleus_sdk::tool_name(&t.endpoint, &t.action),
            description: t.description.clone(),
            params: t.params.to_string(),
        };
        match services.iter_mut().find(|s| s.endpoint == t.endpoint.as_str()) {
            Some(service) => service.tools.push(summary),
            None => services.push(ServiceSummary {
                endpoint: t.endpoint.as_str().to_owned(),
                description: t.endpoint_description.clone(),
                tools: vec![summary],
            }),
        }
    }
    services
}

/// Lean's verdict on one file as the template sees it, bounded to
/// [`PROMPT_LIST_LIMIT`] diagnostics of each kind and [`PROMPT_EXCERPT_CHARS`]
/// per text, with the true counts beside the bounded lists. `invalid_reason`
/// is present (possibly `null`) so a strict template can test it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct CheckSummary {
    pub status: &'static str,
    pub invalid_reason: Option<String>,
    pub errors: Vec<ErrorAt>,
    pub sorries: Vec<GoalAt>,
    /// How many errors Lean reported; `errors` lists the first ones.
    pub errors_total: usize,
    /// How many `sorry`s remain; `sorries` lists the first ones.
    pub sorries_total: usize,
}

/// One Lean error: 1-based line, 0-based column, message excerpt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ErrorAt {
    pub line: u64,
    pub col: u64,
    pub message: String,
}

/// One remaining `sorry`: its position and the goal excerpt.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct GoalAt {
    pub line: u64,
    pub col: u64,
    pub goal: String,
}

impl CheckSummary {
    pub fn new(check: &lean::CheckResponse) -> Self {
        CheckSummary {
            status: status_name(check.status),
            invalid_reason: check.invalid_reason.as_deref().map(|r| excerpt(r, PROMPT_EXCERPT_CHARS)),
            errors: check
                .errors
                .iter()
                .take(PROMPT_LIST_LIMIT)
                .map(|d| ErrorAt {
                    line: d.line,
                    col: d.column,
                    message: excerpt(&d.message, PROMPT_EXCERPT_CHARS),
                })
                .collect(),
            sorries: check
                .sorries
                .iter()
                .take(PROMPT_LIST_LIMIT)
                .map(|s| GoalAt { line: s.line, col: s.column, goal: excerpt(&s.goal, PROMPT_EXCERPT_CHARS) })
                .collect(),
            errors_total: check.errors.len(),
            sorries_total: check.sorries.len(),
        }
    }
}

/// An earlier continuation from the same parent as the template sees it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SiblingSummary {
    pub node: u64,
    pub status: &'static str,
    /// Lean's counts on that sibling's file.
    pub errors: usize,
    pub sorries: usize,
    /// What it changed relative to the parent (untrusted, bounded).
    pub excerpt: String,
}

/// The newest [`PROMPT_LIST_LIMIT`] siblings, in order, excerpts bounded.
pub fn summarize_siblings(siblings: &[explore::Sibling]) -> Vec<SiblingSummary> {
    let skip = siblings.len().saturating_sub(PROMPT_LIST_LIMIT);
    siblings[skip..]
        .iter()
        .map(|s| SiblingSummary {
            node: s.node,
            status: status_name(s.check.status),
            errors: s.check.errors.len(),
            sorries: s.check.sorries.len(),
            excerpt: excerpt(&s.excerpt, PROMPT_EXCERPT_CHARS),
        })
        .collect()
}

/// The messages of one attempt: the rendered system message, the fixed first
/// user message, then the last `keep` (candidate, verdict) exchanges. Older
/// exchanges are dropped, which keeps the context bounded on long runs. The
/// assistant side of an exchange is what the model actually said (its
/// candidate, or the reply that carried none), so it can see its own mistake.
pub fn build_prompt(
    template: &PromptTemplate,
    vars: &PromptVars<'_>,
    exchanges: &[(String, String)],
    keep: usize,
) -> Result<Vec<Message>, PromptError> {
    let system = template.render(vars)?;
    let mut messages =
        vec![Message::text(Role::System, system), Message::text(Role::User, FIRST_USER_MESSAGE)];
    let skip = exchanges.len().saturating_sub(keep);
    for (said, feedback) in &exchanges[skip..] {
        messages.push(Message::text(Role::Assistant, said.clone()));
        messages.push(Message::text(Role::User, feedback.clone()));
    }
    Ok(messages)
}

/// The body of the last fenced code block in `reply` (the info string, e.g.
/// `lean`, is skipped), or `None` when the reply has no complete fenced block.
pub fn last_fenced_block(reply: &str) -> Option<String> {
    let mut last: Option<&str> = None;
    let mut rest = reply;
    while let Some(start) = rest.find("```") {
        let after = &rest[start + 3..];
        // Skip the info string (e.g. `lean`) up to the end of that line.
        let body_start = after.find('\n').map_or(after.len(), |i| i + 1);
        let body = &after[body_start..];
        let Some(end) = body.find("```") else { break };
        last = Some(&body[..end]);
        rest = &body[end + 3..];
    }
    last.map(|b| b.trim().to_owned())
}

/// Why a reply is rejected before any Lean run: no candidate to check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReplyProblem {
    /// The provider cut the reply at the output cap before a complete candidate
    /// appeared. With a reasoning model this usually means the whole budget went
    /// to thinking and no visible answer was produced.
    Truncated { output_tokens: u64 },
    /// The reply carried no candidate at all (empty, or nothing usable).
    Empty,
}

/// What the tool loop did, on the final record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct AgentStats {
    /// Model calls, tool calls, nudges, continues and compactions together.
    pub steps: u32,
    pub nudges: u32,
    pub compactions: u32,
    /// Whether the model handed its answer over itself with `submit`. False
    /// means the budget ended first and the harness submitted on its behalf.
    pub submitted_by_model: bool,
}

impl AgentStats {
    fn of(end: &agent::LoopEnd) -> Self {
        Self {
            steps: end.steps,
            nudges: end.nudges,
            compactions: end.compactions,
            submitted_by_model: end.submitted.is_some(),
        }
    }
}

pub(crate) type Kernel = Connection<std::io::Stdin, std::io::Stdout>;

// ---------------------------------------------------------------------------
// The worker
// ---------------------------------------------------------------------------

/// The task this process continues, named by the authority it was started
/// with: an `explore submit` capability scoped to `[key, problem, task]`. It
/// performs one `CONTINUE(v)` and submits at most one candidate. `task` is the
/// capability's own resource, which every `explore` call names.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Task {
    pub submit: HandleId,
    pub task: ResourcePath,
    pub problem: String,
    pub task_id: String,
}

/// The tool loop's workspace: the `exec` capability on the `shell` endpoint
/// and its resource. Holding one selects the tool loop; its
/// scope must name a workspace (not the root), because the workspace is where
/// the model's commands run and a root scope would let this process name any.
pub fn shell_workspace(
    handles: &[HandleInfo],
    shell_endpoint: &str,
) -> Result<Option<(HandleId, ResourcePath)>, String> {
    let exec = handles.iter().find(|h| {
        h.endpoint.as_str() == shell_endpoint
            && h.actions.iter().any(|a| a.as_str() == nucleus_shell_interface::ACTION_EXEC)
    });
    let Some(exec) = exec else { return Ok(None) };
    if exec.resource.segments().is_empty() {
        return Err(format!(
            "the `{shell_endpoint} exec` capability must be scoped to one workspace, not the root"
        ));
    }
    Ok(Some((exec.handle, exec.resource.clone())))
}

/// The task named by the process's initial handle table: its `submit`
/// capability on `explore_endpoint`, whose resource must be exactly `[run,
/// problem, task]` ([`explore::task_of`]). Without one, or with another
/// resource, the process is misconfigured and refused at startup.
pub fn detect_task(handles: &[HandleInfo], explore_endpoint: &str) -> Result<Task, String> {
    let submit = handles
        .iter()
        .find(|h| {
            h.endpoint.as_str() == explore_endpoint
                && h.actions.iter().any(|a| a.as_str() == explore::ACTION_SUBMIT)
        })
        .ok_or_else(|| format!("started without an `{explore_endpoint} submit` capability"))?;
    let (_, problem, task_id) = explore::task_of(&submit.resource)
        .map_err(|e| format!("the `{explore_endpoint} submit` capability: {}", e.message))?;
    Ok(Task {
        submit: submit.handle,
        task: submit.resource.clone(),
        problem: problem.to_owned(),
        task_id: task_id.to_owned(),
    })
}

pub(crate) fn excerpt(text: &str, limit: usize) -> String {
    let mut out: String = text.chars().take(limit).collect();
    if text.chars().count() > limit {
        out.push_str(" …");
    }
    out
}

/// Lean's verdict on one file, laid out as text for the feedback turn: the
/// status, every error with its position, every remaining `sorry` with its
/// goal. Bounded like [`CheckSummary`], which it is built from; what the bound
/// cut is counted.
pub fn describe_check(check: &lean::CheckResponse) -> String {
    let summary = CheckSummary::new(check);
    let mut text = format!("status: {}", summary.status);
    if let Some(reason) = &summary.invalid_reason {
        text.push_str(&format!(" ({reason})"));
    }
    text.push_str(&format!("\nerrors: {}", summary.errors_total));
    for d in &summary.errors {
        text.push_str(&format!("\n  - line {}, column {}: {}", d.line, d.col, d.message));
    }
    if summary.errors_total > summary.errors.len() {
        text.push_str(&format!("\n  - … and {} more", summary.errors_total - summary.errors.len()));
    }
    text.push_str(&format!("\nremaining sorries: {}", summary.sorries_total));
    for s in &summary.sorries {
        text.push_str(&format!(
            "\n  - line {}, column {}; goal:\n{}",
            s.line,
            s.col,
            indent(&s.goal, "      ")
        ));
    }
    if summary.sorries_total > summary.sorries.len() {
        text.push_str(&format!("\n  - … and {} more", summary.sorries_total - summary.sorries.len()));
    }
    text
}

fn indent(text: &str, prefix: &str) -> String {
    text.lines().map(|l| format!("{prefix}{l}")).collect::<Vec<_>>().join("\n")
}

/// The status as the model and the records see it.
pub fn status_name(status: lean::CheckStatus) -> &'static str {
    match status {
        lean::CheckStatus::Invalid => "invalid",
        lean::CheckStatus::Errors => "errors",
        lean::CheckStatus::Sorries => "sorries",
        lean::CheckStatus::Complete => "complete",
    }
}

/// The order candidates are kept in: `Complete > Sorries > Errors > Invalid`.
pub fn status_rank(status: lean::CheckStatus) -> u8 {
    match status {
        lean::CheckStatus::Invalid => 0,
        lean::CheckStatus::Errors => 1,
        lean::CheckStatus::Sorries => 2,
        lean::CheckStatus::Complete => 3,
    }
}

/// The candidate to submit out of the rounds so far: the highest status, and
/// on a tie the latest. A round whose check never produced a status (the
/// checker was unavailable) is used only when no round has one, in which case
/// the latest candidate is submitted. `None` when there was no candidate at all.
pub fn pick_best(rounds: &[(String, Option<lean::CheckStatus>)]) -> Option<usize> {
    let mut best: Option<(usize, u8)> = None;
    for (i, (_, status)) in rounds.iter().enumerate() {
        if let Some(status) = status {
            let rank = status_rank(*status);
            if best.is_none_or(|(_, r)| rank >= r) {
                best = Some((i, rank));
            }
        }
    }
    best.map(|(i, _)| i).or_else(|| rounds.len().checked_sub(1))
}

/// Inspects a continuation reply: the candidate is the last fenced block; a
/// reply with no block is a failed attempt, because a continuation must
/// return the complete file.
pub fn judge_continue_reply(
    finish: &FinishReason,
    content: &str,
    output_tokens: u64,
) -> Result<String, ReplyProblem> {
    match last_fenced_block(content) {
        Some(candidate) if !candidate.is_empty() => Ok(candidate),
        _ if *finish == FinishReason::Length => Err(ReplyProblem::Truncated { output_tokens }),
        _ => Err(ReplyProblem::Empty),
    }
}

/// One recorded continuation round (`attempts.jsonl`).
#[derive(Clone, Debug, Serialize)]
pub struct ContinueAttempt {
    pub attempt: u32,
    /// Lean's status, when the round produced a candidate and the checker answered.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<lean::CheckStatus>,
    /// Why there is no status: `truncated`, `no_candidate`, `prompt_template`,
    /// or `checker_<code>`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure: Option<String>,
    pub errors: usize,
    pub sorries: usize,
    /// Excerpt of the checker's verdict (or of the failure), untrusted text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub candidate_chars: usize,
}

impl ContinueAttempt {
    /// The record of a round that never reached the model because the
    /// template failed.
    pub fn template_failure(attempt: u32, error: &PromptError) -> Self {
        ContinueAttempt {
            attempt,
            status: None,
            failure: Some(PROMPT_TEMPLATE_FAILURE.to_owned()),
            errors: 0,
            sorries: 0,
            detail: Some(excerpt(&error.to_string(), 2000)),
            input_tokens: 0,
            output_tokens: 0,
            candidate_chars: 0,
        }
    }
}

/// The final record of a continuation (`result.json`).
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "mode", rename = "continue")]
pub struct ContinueOutcome {
    pub problem: String,
    pub task: String,
    pub parent: u64,
    pub model: String,
    /// Model calls made.
    pub attempts: u32,
    /// `complete`, `submitted`, `max_attempts`, `plan_incomplete`,
    /// `model_budget`, `model_error`, `model_outcome_unknown`,
    /// `prompt_template`, `terminated`.
    pub stopped: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub best_status: Option<lean::CheckStatus>,
    /// True iff `explore submit` was called and accepted.
    pub submitted: bool,
    /// The submitted file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// The model or template error that ended the loop, or the submit refusal.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// The tool loop's counters; absent for the fixed loop.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentStats>,
}

/// The worker's `explore assignment` call.
fn fetch_assignment(
    kernel: &mut Kernel,
    handle: HandleId,
    task: &ResourcePath,
    deadline_ms: u64,
) -> Result<explore::Assignment, String> {
    let reply = kernel
        .call(handle, explore::ACTION_ASSIGNMENT, task, &serde_json::json!({}), Some(deadline_ms))
        .map_err(|e| e.to_string())?;
    reply
        .and_then(|p| p.decode::<explore::Assignment>())
        .map_err(|e| format!("cannot read the assignment ({:?}): {}", e.code, e.message))
}

/// The worker's one `explore submit` call.
fn submit_candidate(
    kernel: &mut Kernel,
    handle: HandleId,
    task: &ResourcePath,
    submission: &explore::Submission,
    deadline_ms: u64,
) -> Result<Result<(), Error>, ChannelError> {
    let reply = kernel.call(handle, explore::ACTION_SUBMIT, task, submission, Some(deadline_ms))?;
    Ok(reply.map(|_| ()))
}

/// The fixed loop of one continuation. Generic over the channel only so a
/// test can play the kernel's side; the process runs it on [`Kernel`].
struct Worker<'a, R: Read, W: Write> {
    kernel: Connection<R, W>,
    options: &'a Options,
    model: ModelAccess,
    model_resource: ResourcePath,
    check: HandleId,
    /// The `lean check` capability's own resource, `[key, problem, task]`
    /// (`key` the run's). The `lean` service takes the frozen statement from
    /// it.
    check_resource: ResourcePath,
    assignment: HandleId,
    submit: HandleId,
    /// `[key, problem, task]`, the `explore` capability's own resource.
    task: ResourcePath,
    attempts_log: std::fs::File,
}

/// Where a continuation loop stopped and what it accumulated.
struct LoopEnd {
    stopped: &'static str,
    error: Option<String>,
    calls: u32,
    input_tokens: u64,
    output_tokens: u64,
    /// The provider's own charge for those calls, in nano-USD, summed as they
    /// report it (model interface, `Usage::cost_nano_usd`).
    cost_nano_usd: u64,
    rounds: Vec<(String, Option<lean::CheckStatus>)>,
}

impl<R: Read, W: Write> Worker<'_, R, W> {
    fn record(&mut self, attempt: &ContinueAttempt) -> Result<(), String> {
        let line = serde_json::to_string(attempt).map_err(|e| e.to_string())?;
        writeln!(self.attempts_log, "{line}").map_err(|e| e.to_string())
    }

    /// Deadline for the two `explore` calls. The harness serves `explore` from
    /// the one thread that also runs Lean on the other workers' submissions,
    /// so a call may wait behind a cold `verify`; the kernel's default
    /// deadline (30 s) is too short for that, and a timed-out `submit` would
    /// lose this continuation's candidate.
    fn explore_deadline(&self) -> Option<u64> {
        Some(self.options.verify_deadline_ms)
    }

    fn fetch_assignment(&mut self) -> Result<explore::Assignment, String> {
        let deadline = self.explore_deadline();
        let reply = self
            .kernel
            .call(self.assignment, explore::ACTION_ASSIGNMENT, &self.task, &serde_json::json!({}), deadline)
            .map_err(|e| e.to_string())?;
        reply
            .and_then(|p| p.decode::<explore::Assignment>())
            .map_err(|e| format!("cannot read the assignment ({:?}): {}", e.code, e.message))
    }

    /// `lean check` of one candidate: the candidate alone, on the
    /// capability's own resource, from which the `lean` service takes the
    /// frozen statement. No worker sends frozen text.
    fn check(&mut self, candidate: &str) -> Result<Result<lean::CheckResponse, Error>, ChannelError> {
        let request = lean::CheckRequest { candidate: candidate.to_owned() };
        let reply = self.kernel.call(
            self.check,
            lean::ACTION_CHECK,
            &self.check_resource,
            &request,
            Some(self.options.verify_deadline_ms),
        )?;
        Ok(reply.and_then(|p| p.decode::<lean::CheckResponse>()))
    }

    fn submit(&mut self, submission: &explore::Submission) -> Result<Result<(), Error>, ChannelError> {
        let deadline = self.explore_deadline();
        let reply =
            self.kernel.call(self.submit, explore::ACTION_SUBMIT, &self.task, submission, deadline)?;
        Ok(reply.map(|_| ()))
    }

    /// The persistent model→check loop of one continuation. It ends on a
    /// `Complete` candidate, on the round budget, on a denied or exhausted
    /// model capability, on a template that cannot produce a prompt, or on a
    /// terminate request; it never submits.
    fn explore(&mut self, assignment: &explore::Assignment) -> Result<LoopEnd, String> {
        let budget = u64::from(self.options.max_attempts).min(assignment.model_calls);
        let mut end = LoopEnd {
            stopped: "max_attempts",
            error: None,
            calls: 0,
            input_tokens: 0,
            output_tokens: 0,
            cost_nano_usd: 0,
            rounds: Vec::new(),
        };
        // The assignment's template is the only prompt there is: an empty or
        // broken one fails the first round, recorded, and ends the loop.
        let template = match PromptTemplate::new(&assignment.prompt_template) {
            Ok(t) => t,
            Err(e) => {
                self.record(&ContinueAttempt::template_failure(1, &e))?;
                end.stopped = PROMPT_TEMPLATE_FAILURE;
                end.error = Some(excerpt(&e.to_string(), 2000));
                return Ok(end);
            }
        };
        let check = assignment.check.as_ref().map(CheckSummary::new);
        let siblings = summarize_siblings(&assignment.siblings);
        let file_name = format!("{}.lean", assignment.problem);
        let mut exchanges: Vec<(String, String)> = Vec::new();
        while u64::from(end.calls) < budget {
            if self.kernel.terminate_requested() {
                end.stopped = "terminated";
                return Ok(end);
            }
            let attempt = end.calls + 1;
            let vars = PromptVars {
                mode: "continue",
                problem: &assignment.problem,
                statement: &assignment.statement,
                file: &assignment.file,
                parent: Some(assignment.parent),
                inherited: false,
                check: check.as_ref(),
                siblings: &siblings,
                siblings_total: assignment.siblings.len(),
                branch_references: &[],
                calls_left: budget - u64::from(end.calls),
                attempt,
                file_name: &file_name,
                services: &[],
                publish_plan_tool: None,
                read_plans_tool: None,
                exec_tool: None,
                bash_timeout_ms: None,
            };
            let messages = match build_prompt(&template, &vars, &exchanges, KEEP_EXCHANGES) {
                Ok(m) => m,
                Err(e) => {
                    self.record(&ContinueAttempt::template_failure(attempt, &e))?;
                    end.stopped = PROMPT_TEMPLATE_FAILURE;
                    end.error = Some(excerpt(&e.to_string(), 2000));
                    return Ok(end);
                }
            };
            let response = match complete(
                &mut self.kernel,
                self.options,
                self.model,
                &self.model_resource,
                messages,
                Vec::new(),
                None,
                &mut |_| {},
            )
            .map_err(|e| e.to_string())?
            {
                Ok(r) => r,
                Err(e) => {
                    // The budget speaking (kernel-enforced), or the provider failing:
                    // either way the loop ends and the best so far is submitted.
                    end.stopped =
                        if matches!(e.code, ErrorCode::PermissionDenied | ErrorCode::BudgetExhausted) {
                            "model_budget"
                        } else {
                            "model_error"
                        };
                    end.error = Some(format!("{:?}: {}", e.code, e.message));
                    return Ok(end);
                }
            };
            end.calls += 1;
            end.input_tokens += response.usage.input_tokens;
            end.output_tokens += response.usage.output_tokens;
            end.cost_nano_usd = end.cost_nano_usd.saturating_add(response.usage.cost_nano_usd);
            let content = response.message.content;
            let mut record = ContinueAttempt {
                attempt: end.calls,
                status: None,
                failure: None,
                errors: 0,
                sorries: 0,
                detail: None,
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
                candidate_chars: 0,
            };
            let (said, feedback) = match judge_continue_reply(
                &response.finish_reason,
                &content,
                response.usage.output_tokens,
            ) {
                Ok(candidate) => {
                    record.candidate_chars = candidate.chars().count();
                    let (status, feedback) = match self.check(&candidate).map_err(|e| e.to_string())? {
                        Ok(check) => {
                            record.status = Some(check.status);
                            record.errors = check.errors.len();
                            record.sorries = check.sorries.len();
                            let described = describe_check(&check);
                            record.detail = Some(excerpt(&described, 2000));
                            (Some(check.status), check_feedback(&check, &described))
                        }
                        Err(e) => {
                            record.failure = Some(format!("checker_{:?}", e.code).to_lowercase());
                            record.detail = Some(excerpt(&e.message, 2000));
                            (
                                None,
                                format!(
                                    "The checker could not check that file ({:?}: {}). Return the complete \
                                         file again in one ```lean block.",
                                    e.code,
                                    excerpt(&e.message, 1000)
                                ),
                            )
                        }
                    };
                    end.rounds.push((candidate.clone(), status));
                    let said = format!("```lean\n{candidate}\n```");
                    if status == Some(lean::CheckStatus::Complete) {
                        self.record(&record)?;
                        end.stopped = "complete";
                        return Ok(end);
                    }
                    (said, feedback)
                }
                Err(ReplyProblem::Truncated { output_tokens: n }) => {
                    record.failure = Some("truncated".to_owned());
                    (
                        said_without_candidate(&content),
                        format!(
                            "Your reply was cut off after {n} output tokens before a complete ```lean block \
                                 appeared. Put the complete file FIRST, in one ```lean block, with no explanation."
                        ),
                    )
                }
                Err(ReplyProblem::Empty) => {
                    record.failure = Some("no_candidate".to_owned());
                    (
                        said_without_candidate(&content),
                        "Your reply contained no ```lean block. Return the complete file in one ```lean \
                             block, and nothing else."
                            .to_owned(),
                    )
                }
            };
            self.record(&record)?;
            exchanges.push((said, feedback));
        }
        Ok(end)
    }
}

/// The assistant turn echoed back when a reply carried no candidate: the reply
/// itself (bounded), or a placeholder when it was empty, so the conversation
/// never contains an empty assistant message.
fn said_without_candidate(content: &str) -> String {
    let said = excerpt(content, 2000);
    if said.trim().is_empty() { "(empty reply)".to_owned() } else { said }
}

/// The user turn after a checked candidate: the verdict, then what to do next
/// ("at least one" error or `sorry`, never exactly one, so the model is not
/// told to stop after a single fix).
fn check_feedback(check: &lean::CheckResponse, described: &str) -> String {
    let next = match check.status {
        lean::CheckStatus::Invalid => {
            "That is not a valid attempt at this problem. Keep the theorem statement exactly as given above, \
             byte for byte, and do not use `native_decide` or `axiom`."
        }
        lean::CheckStatus::Errors => "Fix at least one error and return the complete file.",
        lean::CheckStatus::Sorries => {
            "Close or decompose at least one remaining `sorry` (more if you can) and return the complete file."
        }
        lean::CheckStatus::Complete => "Return the complete file.",
    };
    format!("Checked:\n{described}\n\n{next}")
}

/// One `CONTINUE(v)`: fetch the assignment, run the loop, submit the best
/// candidate when there is one, write the records. A refused submission, or a template that produced no
/// prompt, is reported in `result.json` and as this function's error, so the
/// exit status says so.
fn run_continue(
    kernel: Kernel,
    options: &Options,
    model: ModelAccess,
    submit: HandleId,
    task: ResourcePath,
    problem_id: String,
    task_id: String,
) -> Result<(), String> {
    if let Some((shell, workspace)) = shell_workspace(kernel.handles(), &options.shell_endpoint)? {
        return run_continue_agent(
            kernel, options, model, submit, task, shell, workspace, problem_id, task_id,
        );
    }
    let check = kernel
        .find_handle(&options.lean_endpoint, lean::ACTION_CHECK)
        .ok_or("the worker needs a `lean check` capability")?;
    let assignment = kernel
        .find_handle(&options.explore_endpoint, explore::ACTION_ASSIGNMENT)
        .ok_or("the worker needs an `explore assignment` capability")?;
    let out = options.out.join(&problem_id).join(&task_id);
    std::fs::create_dir_all(&out).map_err(|e| format!("out dir: {e}"))?;
    let attempts_log = std::fs::File::create(out.join("attempts.jsonl")).map_err(|e| e.to_string())?;
    let model_resource = model::resource_for(&options.model).map_err(|e| e.to_string())?;
    // The capability's own resource, `[key, problem, task]`, where the lean
    // service keeps the run's registered statement.
    let check_resource = kernel
        .handles()
        .iter()
        .find(|h| h.handle == check)
        .map(|h| h.resource.clone())
        .ok_or("the check handle is not in this process's handle table")?;
    let mut worker = Worker {
        kernel,
        options,
        model,
        model_resource,
        check,
        check_resource,
        assignment,
        submit,
        task,
        attempts_log,
    };
    let assignment = worker.fetch_assignment()?;
    if assignment.problem != problem_id || assignment.task != task_id {
        return Err(format!(
            "the assignment is for {}/{} but this worker's authority names {problem_id}/{task_id}",
            assignment.problem, assignment.task
        ));
    }
    let end = worker.explore(&assignment)?;
    let best = pick_best(&end.rounds);
    let mut outcome = ContinueOutcome {
        problem: problem_id,
        task: task_id,
        parent: assignment.parent,
        model: options.model.clone(),
        attempts: end.calls,
        stopped: end.stopped,
        best_status: best.and_then(|i| end.rounds[i].1),
        submitted: false,
        file: None,
        error: end.error.clone(),
        input_tokens: end.input_tokens,
        output_tokens: end.output_tokens,
        agent: None,
    };
    let mut failure = None;
    if let Some(i) = best {
        let file = end.rounds[i].0.clone();
        let submission = explore::Submission {
            file: file.clone(),
            model_calls: end.calls,
            input_tokens: end.input_tokens,
            output_tokens: end.output_tokens,
            cost_nano_usd: end.cost_nano_usd,
            // The budget ended; there is no later attempt to improve this.
            final_offer: true,
        };
        match worker.submit(&submission).map_err(|e| e.to_string())? {
            Ok(()) => {
                outcome.submitted = true;
                outcome.file = Some(file);
            }
            Err(e) => {
                let why = format!("submit refused ({:?}): {}", e.code, e.message);
                outcome.error = Some(why.clone());
                failure = Some(why);
            }
        }
    }
    if failure.is_none() && end.stopped == PROMPT_TEMPLATE_FAILURE {
        failure = end.error.clone();
    }
    let json = serde_json::to_string_pretty(&outcome).map_err(|e| e.to_string())?;
    std::fs::write(out.join("result.json"), json).map_err(|e| e.to_string())?;
    failure.map_or(Ok(()), Err)
}

/// Performs one continuation and writes the outcome; exits when done. A run
/// that ended because the template produced no prompt is written like any
/// other and then reported as this function's error.
pub fn run(options: &Options) -> Result<(), String> {
    let kernel = Connection::from_stdio().map_err(|e| e.to_string())?;
    // Streaming needs `start` on the budgeted handle and `next`/`cancel` on
    // an unbudgeted one; without both, the loop blocks on `complete` and
    // cannot tell a thinking provider from a stalled one.
    let handle = kernel
        .find_handle(model::ENDPOINT, model::ACTION_COMPLETE)
        .ok_or("started without a `model complete` capability")?;
    let can_start = kernel
        .handles()
        .iter()
        .any(|h| h.handle == handle && h.actions.iter().any(|a| a.as_str() == model::ACTION_START));
    let pulls = kernel.find_handle(model::ENDPOINT, model::ACTION_NEXT).filter(|_| can_start);
    let model = ModelAccess { handle, pulls };
    let Task { submit, task, problem, task_id } = detect_task(kernel.handles(), &options.explore_endpoint)?;
    run_continue(kernel, options, model, submit, task, problem, task_id)
}

/// Renders a tool-loop message; an empty render is a
/// template failure like any other (there is no built-in prompt).
fn render_mode(
    template: &PromptTemplate,
    vars: &PromptVars<'_>,
    mode: &'static str,
) -> Result<String, String> {
    let vars = PromptVars { mode, ..vars.clone() };
    let text = template.render(&vars).map_err(|e| e.to_string())?;
    if text.trim().is_empty() {
        return Err(format!("{PROMPT_TEMPLATE_FAILURE} rendered nothing for mode {mode:?}"));
    }
    Ok(text)
}

/// One `CONTINUE(v)` with the tool loop: from the parent's transcript when
/// the assignment names one, else a fresh conversation on the parent's file.
/// What the model submits is the node's file; when it submits nothing, the
/// workspace file at the end is submitted on its behalf.
#[allow(clippy::too_many_arguments)]
fn run_continue_agent(
    mut kernel: Kernel,
    options: &Options,
    model: ModelAccess,
    submit: HandleId,
    task: ResourcePath,
    shell: HandleId,
    workspace: ResourcePath,
    problem_id: String,
    task_id: String,
) -> Result<(), String> {
    let tools = kernel.tools().map_err(|e| e.to_string())?.map_err(|e| format!("tools: {}", e.message))?;
    let services = summarize_services(&tools);
    let exec_tool =
        agent_prompts::listed_tool(&tools, &options.shell_endpoint, nucleus_shell_interface::ACTION_EXEC);
    let assignment_handle = kernel
        .find_handle(&options.explore_endpoint, explore::ACTION_ASSIGNMENT)
        .ok_or("the worker needs an `explore assignment` capability")?;
    let out = options.out.join(&problem_id).join(&task_id);
    std::fs::create_dir_all(&out).map_err(|e| format!("out dir: {e}"))?;
    let model_resource = model::resource_for(&options.model).map_err(|e| e.to_string())?;
    let assignment = fetch_assignment(&mut kernel, assignment_handle, &task, options.verify_deadline_ms)?;
    if assignment.problem != problem_id || assignment.task != task_id {
        return Err(format!(
            "the assignment is for {}/{} but this worker's authority names {problem_id}/{task_id}",
            assignment.problem, assignment.task
        ));
    }
    let budget = u64::from(options.max_attempts).min(assignment.model_calls);
    let mut outcome = ContinueOutcome {
        problem: problem_id.clone(),
        task: task_id.clone(),
        parent: assignment.parent,
        model: options.model.clone(),
        attempts: 0,
        stopped: PROMPT_TEMPLATE_FAILURE,
        best_status: None,
        submitted: false,
        file: None,
        error: None,
        input_tokens: 0,
        output_tokens: 0,
        agent: None,
    };
    let check = assignment.check.as_ref().map(CheckSummary::new);
    let siblings = summarize_siblings(&assignment.siblings);
    let file_name = format!("{problem_id}.lean");
    let publish_plan_tool =
        agent_prompts::listed_tool(&tools, &options.explore_endpoint, explore::ACTION_PUBLISH_PLAN);
    let read_plans_tool =
        agent_prompts::listed_tool(&tools, &options.explore_endpoint, explore::ACTION_READ_PLANS);
    let vars = PromptVars {
        mode: "system",
        problem: &problem_id,
        statement: &assignment.statement,
        file: &assignment.file,
        parent: Some(assignment.parent),
        inherited: assignment.parent_transcript.is_some(),
        check: check.as_ref(),
        siblings: &siblings,
        siblings_total: assignment.siblings.len(),
        branch_references: &assignment.branch_references,
        calls_left: budget,
        attempt: 1,
        file_name: &file_name,
        services: &services,
        publish_plan_tool: publish_plan_tool.as_deref(),
        read_plans_tool: read_plans_tool.as_deref(),
        exec_tool: exec_tool.as_deref(),
        bash_timeout_ms: Some(options.bash_timeout_ms),
    };
    let start =
        PromptTemplate::new(&assignment.prompt_template).map_err(|e| e.to_string()).and_then(|template| {
            let prompts =
                agent_prompts::AgentPrompts::render(&template, &vars, &tools, &options.explore_endpoint)?;
            let start = match &assignment.parent_transcript {
                Some(path) => agent::Start::Continued {
                    system: render_mode(&template, &vars, "system")?,
                    parent_transcript: PathBuf::from(path),
                    resume: render_mode(&template, &vars, "resume")?,
                },
                // A continuation of the root has no transcript to inherit,
                // but the harness still knows what its siblings tried and what
                // the node's file checks as: the `resume` message carries it.
                None => agent::Start::Fresh {
                    system: render_mode(&template, &vars, "system")?,
                    task: render_mode(&template, &vars, "task")?,
                    resume: render_mode(&template, &vars, "resume")?,
                },
            };
            Ok((start, prompts))
        });
    let mut failure: Option<String> = None;
    match start {
        Err(why) => failure = Some(why),
        Ok((start, prompts)) => {
            let end = {
                let mut agent = agent::Agent::open(
                    &mut kernel,
                    options,
                    model,
                    model_resource,
                    shell,
                    workspace,
                    tools,
                    &problem_id,
                    Some(&task_id),
                    &out,
                    &assignment.statement,
                    &assignment.file,
                    start,
                    prompts,
                )?;
                agent.write_branch_index(&assignment.branch_references)?;
                agent.run(budget)?
            };
            outcome.attempts = end.calls;
            outcome.stopped = end.stopped;
            outcome.error = end.error.clone();
            outcome.input_tokens = end.input_tokens;
            outcome.output_tokens = end.output_tokens;
            outcome.best_status = end.complete.then_some(lean::CheckStatus::Complete);
            outcome.agent = Some(AgentStats::of(&end));
            if matches!(end.stopped, "model_outcome_unknown" | "plan_incomplete") {
                // No child: a stream whose release was not confirmed may
                // still be running (nothing retries it), and a worker that
                // only planned did no work. The record says which; the
                // harness counts the continuation as failed.
                outcome.file = end.file.clone();
                write_result(&out, &outcome)?;
                return Ok(());
            }
            // What the loop did is known now; whether the harness takes the
            // submission is not. Write the record before asking, because the
            // harness may end the round the moment it answers and this
            // process is its child: the reply waits for the Lean run, so
            // there is no window between "accepted" and "killed" to write
            // anything in.
            write_result(&out, &outcome)?;
            match (end.submitted.clone(), end.file.clone()) {
                // The model called `explore submit` itself: the harness has
                // the file it chose, and there is nothing left to send — a
                // second submission would be refused, and rightly.
                (Some(file), _) => {
                    outcome.submitted = true;
                    outcome.file = Some(file);
                }
                (None, None) => {
                    failure = Some("the workspace file could not be read at the end of the loop".to_owned())
                }
                // The budget (or an error) ended the loop before the model
                // handed anything over: the workspace file is what this node
                // is, and the record says the harness sent it
                // (`submitted_by_model: false`).
                (None, Some(file)) => {
                    let submission = explore::Submission {
                        file: file.clone(),
                        model_calls: end.calls,
                        input_tokens: end.input_tokens,
                        output_tokens: end.output_tokens,
                        cost_nano_usd: end.cost_nano_usd,
                        // The harness submitting on the model's behalf: last word.
                        final_offer: true,
                    };
                    match submit_candidate(
                        &mut kernel,
                        submit,
                        &task,
                        &submission,
                        options.verify_deadline_ms,
                    )
                    .map_err(|e| e.to_string())?
                    {
                        Ok(()) => {
                            outcome.submitted = true;
                            outcome.file = Some(file);
                        }
                        Err(e) => failure = Some(format!("submit refused ({:?}): {}", e.code, e.message)),
                    }
                }
            }
        }
    }
    // An error the loop ended on stays the record's reason: after a model
    // error the harness offers the unchanged file, and the refusal of that
    // offer says nothing about why the worker stopped.
    if let Some(why) = &failure
        && outcome.error.is_none()
    {
        outcome.error = Some(excerpt(why, 2000));
    }
    write_result(&out, &outcome)?;
    failure.map_or(Ok(()), Err)
}

/// The worker's own record of its attempt, rewritten whenever more of it is
/// known. Written at least once before the submission, so a worker the
/// harness kills on the round's last breath still leaves its evidence.
fn write_result(out: &Path, outcome: &ContinueOutcome) -> Result<(), String> {
    let json = serde_json::to_string_pretty(outcome).map_err(|e| e.to_string())?;
    std::fs::write(out.join("result.json"), json).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::collections::BTreeSet;

    use lean::{CheckResponse, CheckStatus, Diagnostic, OpenGoal};
    use nucleus_protocol::{ActionName, EndpointName};

    use super::*;

    /// The template that ships with the prover.
    const DEFAULT_TEMPLATE: &str = include_str!("../prompts/putnam.j2");

    fn default_template() -> PromptTemplate {
        PromptTemplate::new(DEFAULT_TEMPLATE).unwrap()
    }

    #[test]
    fn options_require_model_and_out() {
        let parse = |extra: &[&str]| {
            Options::parse(["--model", "m", "--out", "/o"].iter().chain(extra).map(|s| (*s).to_owned()))
        };
        assert!(Options::parse(["--model", "m"].map(str::to_owned)).is_err(), "--out is required");
        // Extra arguments, and what they must yield.
        type Accepted = (&'static [&'static str], fn(&Options) -> bool);
        let accepted: &[Accepted] = &[
            (&[], |o| {
                (o.max_attempts, o.reasoning_effort, o.temperature) == (8, None, None)
                    && (o.model_retries, o.model_retry_backoff_ms) == (3, 2000)
                    && o.explore_endpoint == "explore"
            }),
            (&["--max-attempts", "3"], |o| o.max_attempts == 3),
            (&["--reasoning-effort", "low"], |o| o.reasoning_effort == Some(ReasoningEffort::Low)),
            (&["--reasoning-effort", "off"], |o| o.reasoning_effort == Some(ReasoningEffort::Off)),
            (&["--reasoning-effort", "max"], |o| o.reasoning_effort == Some(ReasoningEffort::Max)),
            (&["--model-retries", "0", "--model-retry-backoff-ms", "0"], |o| {
                (o.model_retries, o.model_retry_backoff_ms) == (0, 0)
            }),
            (&["--temperature", "0"], |o| o.temperature == Some(0.0)),
            (&["--temperature", "2"], |o| o.temperature == Some(2.0)),
            (&["--explore-endpoint", "tree"], |o| o.explore_endpoint == "tree"),
        ];
        for (extra, holds) in accepted {
            assert!(holds(&parse(extra).unwrap()), "{extra:?}");
        }
        let refused: &[&[&str]] = &[
            &["--max-attempts", "0"],
            &["--reasoning-effort", "unknown"],
            &["--model-retry-backoff-ms", "60001"],
            &["--model-retries", "-1"],
            &["--temperature", "NaN"],
            &["--temperature", "inf"],
            &["--temperature", "-0.1"],
            &["--temperature", "2.1"],
            &["--temperature", "invalid"],
        ];
        for extra in refused {
            assert!(parse(extra).is_err(), "{extra:?}");
        }
    }

    #[test]
    fn the_last_lean_block_is_the_candidate() {
        let reply = "Thinking...\n```lean\nfirst\n```\nActually:\n```lean\nimport Mathlib\ntheorem t : True := trivial\n```\nDone.";
        assert_eq!(last_fenced_block(reply).as_deref(), Some("import Mathlib\ntheorem t : True := trivial"));
        assert_eq!(last_fenced_block("no fences here"), None);
        assert_eq!(last_fenced_block("```\nbare fence\n```").as_deref(), Some("bare fence"));
    }

    // ----- The system-prompt template -----

    const STATEMENT: &str = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n :=\nsorry\n";

    fn vars<'a>(attempt: u32, calls_left: u64) -> PromptVars<'a> {
        PromptVars {
            mode: "continue",
            problem: "demo_thm",
            statement: STATEMENT,
            file: STATEMENT,
            parent: None,
            inherited: false,
            check: None,
            siblings: &[],
            siblings_total: 0,
            branch_references: &[],
            calls_left,
            attempt,
            file_name: "demo_thm.lean",
            services: &[],
            publish_plan_tool: None,
            read_plans_tool: None,
            exec_tool: None,
            bash_timeout_ms: None,
        }
    }

    #[test]
    fn the_template_variables_are_fixed_and_the_default_template_ignores_the_attempt_counters() {
        // What a template can reference: the serialized keys of `PromptVars`
        // are exactly these; `file_name`, the services, the branch references,
        // the plan tools and the command limit serve the tool loop.
        let keys = |value: &serde_json::Value| -> BTreeSet<String> {
            value.as_object().unwrap().keys().cloned().collect()
        };
        let names = |list: &[&str]| -> BTreeSet<String> { list.iter().map(|s| (*s).to_owned()).collect() };
        assert_eq!(
            keys(&serde_json::to_value(vars(1, 8)).unwrap()),
            names(&[
                "inherited",
                "mode",
                "problem",
                "statement",
                "file",
                "parent",
                "check",
                "siblings",
                "siblings_total",
                "branch_references",
                "calls_left",
                "attempt",
                "file_name",
                "services",
                "publish_plan_tool",
                "read_plans_tool",
                "exec_tool",
                "bash_timeout_ms"
            ])
        );
        let a = assignment();
        let verdict = a.check.as_ref().map(CheckSummary::new);
        assert_eq!(
            keys(&serde_json::to_value(verdict.as_ref().unwrap()).unwrap()),
            names(&["status", "invalid_reason", "errors", "sorries", "errors_total", "sorries_total"])
        );
        assert_eq!(
            keys(&serde_json::to_value(&summarize_siblings(&a.siblings)[0]).unwrap()),
            names(&["node", "status", "errors", "sorries", "excerpt"])
        );

        // The default template's system message does not depend on the
        // attempt counters: a worker's prompt prefix is stable across attempts.
        let template = default_template();
        let siblings = summarize_siblings(&a.siblings);
        let first = template.render(&continue_vars(&a, verdict.as_ref(), &siblings, 1)).unwrap();
        let third = template.render(&continue_vars(&a, verdict.as_ref(), &siblings, 3)).unwrap();
        assert_eq!(first, third);
        assert!(!first.contains("call(s) left") && !first.contains("Attempt "), "{first}");
        let body = DEFAULT_TEMPLATE.split_once("-#}").map_or(DEFAULT_TEMPLATE, |(_, body)| body);
        assert!(!body.contains("{{ attempt") && !body.contains("calls_left"), "{body}");
    }

    #[test]
    fn a_runaway_template_fails_the_attempt_instead_of_pinning_the_worker() {
        let vars = vars(1, 1);
        // Nested loops that would run for hours: the fuel runs out first.
        let looping = PromptTemplate::new(
            "{% for i in range(100000) %}{% for j in range(100000) %}{% set x = i * j %}{% endfor %}{% endfor %}",
        )
        .unwrap();
        let started = std::time::Instant::now();
        match looping.render(&vars) {
            Err(PromptError::Render(e)) => assert!(e.contains("fuel"), "{e}"),
            other => panic!("a looping template must fail: {other:?}"),
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(30), "{:?}", started.elapsed());
        // Output past the bound fails too, even within the fuel.
        let printing =
            PromptTemplate::new("{% for i in range(3000) %}{{ statement }}\n{% endfor %}").unwrap();
        match printing.render(&vars) {
            Err(PromptError::Render(e)) => assert!(e.contains("exceeds"), "{e}"),
            other => panic!("an over-long render must fail: {other:?}"),
        }
        // The default template is nowhere near either bound.
        assert!(default_template().render(&vars).is_ok());
        assert!(default_template().render(&vars).unwrap().len() < 4096);
    }

    #[test]
    fn an_undefined_variable_or_an_include_is_an_error_and_nothing_is_escaped() {
        let vars = vars(1, 1);
        let strict = PromptTemplate::new("{{ problem }} {{ nope }}").unwrap();
        match strict.render(&vars) {
            Err(PromptError::Render(e)) => assert!(e.contains("nope"), "{e}"),
            other => panic!("strict undefined must fail: {other:?}"),
        }
        // A key missing from a defined object fails too; a present `none` does not.
        assert!(PromptTemplate::new("{{ check.status }}").unwrap().render(&vars).is_err(), "check is none");
        assert!(PromptTemplate::new("{% if check %}x{% endif %}").unwrap().render(&vars).is_ok());
        // No other template exists and no loader is configured: includes cannot work.
        assert!(matches!(PromptTemplate::new("{% include 'other.j2' %}"), Err(PromptError::Syntax(_))));
        assert!(matches!(PromptTemplate::new("{% extends 'base.j2' %}"), Err(PromptError::Syntax(_))));
        // Values are rendered as text: no HTML escaping of Lean.
        let lean = "theorem t (a b : Nat) (h : a < b ∧ b ≤ a + 1) : ¬ (a ≥ b) := by omega -- <not & 'html'>";
        let raw = PromptTemplate::new("{{ statement }}").unwrap();
        let rendered = raw.render(&PromptVars { statement: lean, ..vars }).unwrap();
        assert_eq!(rendered, lean);
    }

    #[test]
    fn an_empty_template_fails_the_attempt_with_a_recorded_reason() {
        for empty in ["", "   ", "\n\t\n"] {
            assert_eq!(PromptTemplate::new(empty).err(), Some(PromptError::Empty));
        }
        // An assignment with an empty template fails its first round before
        // any model call…
        let assignment = explore::Assignment { prompt_template: String::new(), ..assignment() };
        let failure = PromptTemplate::new(&assignment.prompt_template).unwrap_err();
        assert_eq!(failure.to_string(), "prompt_template is empty");
        // …and the round is recorded with that reason, no tokens spent.
        let record = serde_json::to_value(ContinueAttempt::template_failure(1, &failure)).unwrap();
        assert_eq!(record["failure"], PROMPT_TEMPLATE_FAILURE);
        assert_eq!(record["detail"], "prompt_template is empty");
        assert_eq!(record["attempt"].as_u64(), Some(1));
        assert_eq!((record["input_tokens"].as_u64(), record["output_tokens"].as_u64()), (Some(0), Some(0)));
        assert!(record.get("status").is_none());
    }

    // ----- the worker -----

    fn handle(id: u32, endpoint: &str, actions: &[&str], resource: &[&str]) -> HandleInfo {
        HandleInfo {
            handle: HandleId(id),
            endpoint: EndpointName::new(endpoint).unwrap(),
            actions: actions.iter().map(|a| ActionName::new(*a).unwrap()).collect::<BTreeSet<_>>(),
            resource: ResourcePath::new(resource.iter().copied()).unwrap(),
            delegable: false,
            expires_in_ms: None,
            max_calls: None,
        }
    }

    #[test]
    fn the_task_is_named_by_an_explore_submit_capability() {
        let model = handle(1, "model", &["complete"], &["deepseek"]);
        let without = vec![model.clone(), handle(2, "lean", &["statement", "verify"], &["demo_thm"])];
        assert!(detect_task(&without, "explore").is_err(), "no submit capability, no task");

        let worker = vec![
            model.clone(),
            handle(2, "lean", &["check"], &["demo_thm"]),
            handle(3, "explore", &["assignment", "submit"], &["gen2", "demo_thm", "t7"]),
        ];
        assert_eq!(
            detect_task(&worker, "explore"),
            Ok(Task {
                submit: HandleId(3),
                task: ResourcePath::new(["gen2", "demo_thm", "t7"]).unwrap(),
                problem: "demo_thm".into(),
                task_id: "t7".into(),
            })
        );
        // The endpoint name is the configured one, not a hard-wired string.
        assert!(detect_task(&worker, "tree").is_err());
        let renamed = vec![model.clone(), handle(3, "tree", &["submit"], &["r", "p", "t"])];
        assert!(detect_task(&renamed, "tree").is_ok());

        // `assignment` alone does not make a worker: only `submit` does.
        let observer = vec![model.clone(), handle(3, "explore", &["assignment"], &["r", "p", "t"])];
        assert!(detect_task(&observer, "explore").is_err());

        // A submit capability of any other shape, `[problem, task]` included,
        // is a misconfiguration, not a fallback.
        for shape in [&["p", "t"][..], &["p"], &[], &["r", "p", "t", "x"]] {
            let wrong = vec![model.clone(), handle(3, "explore", &["submit"], shape)];
            assert!(detect_task(&wrong, "explore").is_err(), "{shape:?}");
        }
    }

    fn check(status: CheckStatus, errors: &[(u64, &str)], sorries: &[(u64, &str)]) -> CheckResponse {
        CheckResponse {
            status,
            invalid_reason: None,
            errors: errors
                .iter()
                .map(|(line, m)| Diagnostic { line: *line, column: 2, message: (*m).to_owned() })
                .collect(),
            sorries: sorries
                .iter()
                .map(|(line, g)| OpenGoal { line: *line, column: 4, goal: (*g).to_owned() })
                .collect(),
            elapsed_ms: 80,
        }
    }

    fn assignment() -> explore::Assignment {
        explore::Assignment {
            problem: "demo_thm".into(),
            task: "t7".into(),
            parent: 3,
            statement: STATEMENT.into(),
            file: "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n := by\n  have h : 0 = 0 := by exact rfl\n  sorry\n".into(),
            check: Some(check(
                CheckStatus::Errors,
                &[(4, "unknown identifier 'foo'")],
                &[(5, "n : Nat\nh : 0 = 0\n⊢ n + 0 = n")],
            )),
            siblings: vec![
                explore::Sibling {
                    node: 4,
                    check: check(CheckStatus::Sorries, &[], &[(5, "⊢ n = n")]),
                    excerpt: "+  rw [Nat.add_zero]".into(),
                },
                explore::Sibling {
                    node: 5,
                    check: check(CheckStatus::Errors, &[(5, "simp made no progress")], &[]),
                    excerpt: "+  simp".into(),
                },
            ],
            model_calls: 3,
            prompt_template: DEFAULT_TEMPLATE.into(),
            parent_transcript: None,
            branch_references: vec![],
        }
    }

    /// The variables a worker builds from its assignment, as `Worker::explore` does.
    fn continue_vars<'a>(
        a: &'a explore::Assignment,
        check: Option<&'a CheckSummary>,
        siblings: &'a [SiblingSummary],
        attempt: u32,
    ) -> PromptVars<'a> {
        PromptVars {
            mode: "continue",
            inherited: false,
            problem: &a.problem,
            statement: &a.statement,
            file: &a.file,
            parent: Some(a.parent),
            check,
            siblings,
            siblings_total: a.siblings.len(),
            branch_references: &a.branch_references,
            calls_left: a.model_calls - u64::from(attempt) + 1,
            attempt,
            file_name: "demo_thm.lean",
            services: &[],
            publish_plan_tool: None,
            read_plans_tool: None,
            exec_tool: None,
            bash_timeout_ms: None,
        }
    }

    #[test]
    fn a_continuation_from_the_root_shows_the_statement_once_and_says_so() {
        let a = explore::Assignment { file: assignment().statement.clone(), parent: 0, ..assignment() };
        let template = PromptTemplate::new(&a.prompt_template).unwrap();
        let system = template.render(&continue_vars(&a, None, &[], 1)).unwrap();
        assert_eq!(system.matches(&a.statement).count(), 1, "the file is not repeated: {system}");
        assert!(system.contains("The current file is the statement itself"), "{system}");
        assert!(
            system.contains("`let (…) := (…)`") && system.contains("simp only []"),
            "the unfolding hint: {system}"
        );
        let b = assignment();
        let child = template.render(&continue_vars(&b, None, &[], 1)).unwrap();
        assert!(
            child.contains("The current file (the node's workspace):"),
            "a changed file is shown: {child}"
        );
        assert_eq!(
            child.matches(&b.statement).count(),
            1,
            "the parent's file differs, so the statement appears once"
        );
    }

    #[test]
    fn the_default_template_renders_the_continuation_prompt() {
        let a = assignment();
        let template = PromptTemplate::new(&a.prompt_template).unwrap();
        let verdict = a.check.as_ref().map(CheckSummary::new);
        let siblings = summarize_siblings(&a.siblings);
        let messages =
            build_prompt(&template, &continue_vars(&a, verdict.as_ref(), &siblings, 1), &[], 4).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::System);
        assert_eq!((messages[1].role, messages[1].content.as_str()), (Role::User, FIRST_USER_MESSAGE));
        let system = &messages[0].content;
        assert!(system.contains("continuing a partial proof"), "{system}");
        assert!(system.contains("`sorry` is allowed"));
        assert!(system.contains("`native_decide` and `axiom` are not"));
        assert!(system.contains("byte for byte"));
        assert!(system.contains("at least one thing"), "at least one, never exactly one: {system}");
        assert!(system.contains("Do not rewrite parts that already check"));
        assert!(system.contains("continuing from node 3"), "names the parent: {system}");
        assert!(system.contains(&a.statement), "the frozen statement, verbatim");
        assert!(system.contains(&a.file), "the parent's file, verbatim");
        assert!(system.contains("status: errors\nerrors: 1\n"), "{system}");
        assert!(system.contains("line 4, column 2: unknown identifier 'foo'"), "each error with its line");
        assert!(system.contains("remaining sorries: 1\n  - line 5, column 4; goal:\n      n : Nat\n      h : 0 = 0\n      ⊢ n + 0 = n\n"), "each sorry with its position and goal, indented: {system}");
        assert!(system.contains(
            "already made from this same node (2); each was already tried, so do something different:"
        ));
        assert!(
            system.contains(
                "  - node 4: sorries (0 error(s), 1 sorry(ies) remaining)\n    what it changed:\n      +  rw [Nat.add_zero]\n"
            ),
            "{system}"
        );
        assert!(
            system.contains("  - node 5: errors (1 error(s), 0 sorry(ies) remaining)\n    what it changed:\n      +  simp\n"),
            "{system}"
        );
        assert!(!system.contains("omitted"), "nothing was cut: {system}");

        // Previous rounds of this loop follow as (said, verdict) pairs, trimmed to `keep`.
        let exchanges = vec![
            ("```lean\nold\n```".to_owned(), "Checked:\nstatus: errors".to_owned()),
            ("```lean\nnewer\n```".to_owned(), "Checked:\nstatus: sorries".to_owned()),
        ];
        let messages =
            build_prompt(&template, &continue_vars(&a, verdict.as_ref(), &siblings, 3), &exchanges, 1)
                .unwrap();
        assert_eq!(messages.len(), 4);
        assert_eq!(&messages[0].content, system, "the same system message on the third call");
        assert_eq!(
            (messages[2].role, messages[2].content.as_str()),
            (Role::Assistant, "```lean\nnewer\n```")
        );
        assert_eq!(
            (messages[3].role, messages[3].content.as_str()),
            (Role::User, "Checked:\nstatus: sorries")
        );

        // The root has no verdict and no siblings; the prompt says so instead of inventing one.
        let root = explore::Assignment { parent: 0, check: None, siblings: vec![], ..a };
        let system =
            &build_prompt(&template, &continue_vars(&root, None, &[], 1), &[], 4).unwrap()[0].content;
        assert!(system.contains("continuing from node 0"));
        assert!(system.contains("root node: not checked yet"), "{system}");
        assert!(!system.contains("already tried"));

        // An invalid parent shows the reason next to the status.
        let mut invalid = check(CheckStatus::Invalid, &[], &[]);
        invalid.invalid_reason = Some("statement mismatch".into());
        let summary = CheckSummary::new(&invalid);
        let system = &build_prompt(&template, &continue_vars(&root, Some(&summary), &[], 1), &[], 4).unwrap()
            [0]
        .content;
        assert!(system.contains("status: invalid (statement mismatch)\nerrors: 0\n"), "{system}");
    }

    #[test]
    fn the_verdict_handed_to_the_template_is_bounded_and_both_forms_count_the_rest() {
        let long = "x".repeat(PROMPT_EXCERPT_CHARS + 10);
        let limit = u64::try_from(PROMPT_LIST_LIMIT).unwrap();
        let errors: Vec<(u64, &str)> = (1..=limit + 3).map(|i| (i, long.as_str())).collect();
        let sorries: Vec<(u64, &str)> = (1..=2).map(|i| (i, "⊢ True")).collect();
        let c = check(CheckStatus::Errors, &errors, &sorries);
        let summary = CheckSummary::new(&c);
        assert_eq!(summary.errors.len(), PROMPT_LIST_LIMIT);
        assert_eq!(summary.sorries.len(), 2);
        assert_eq!(summary.errors[0].message.chars().count(), PROMPT_EXCERPT_CHARS + 2, "cut, then ` …`");
        assert!(summary.errors[0].message.ends_with(" …"));
        assert_eq!((summary.errors[0].line, summary.errors[0].col), (1, 2));
        assert_eq!(summary.invalid_reason, None, "present as null for a strict template");
        assert_eq!((summary.errors_total, summary.sorries_total), (15, 2), "the true counts travel too");
        let json = serde_json::to_value(&summary).unwrap();
        assert!(json["invalid_reason"].is_null());
        assert_eq!(json["sorries"][0]["col"], 4);

        let text = describe_check(&c);
        assert!(text.contains("errors: 15"), "{text}");
        assert!(text.contains("  - … and 3 more"), "{text}");
        assert!(text.contains("remaining sorries: 2"));

        let many: Vec<explore::Sibling> = (1..=limit + 2)
            .map(|node| explore::Sibling {
                node,
                check: check(CheckStatus::Sorries, &[], &[]),
                excerpt: long.clone(),
            })
            .collect();
        let siblings = summarize_siblings(&many);
        assert_eq!(siblings.len(), PROMPT_LIST_LIMIT);
        assert_eq!(siblings[0].node, 3, "the newest are kept");
        assert_eq!(siblings[0].status, "sorries");
        assert!(siblings[0].excerpt.ends_with(" …"));

        // The default template prints the same totals and omissions as the text form.
        let a = explore::Assignment { check: Some(c), siblings: many, ..assignment() };
        let vars = PromptVars {
            mode: "continue",
            inherited: false,
            problem: &a.problem,
            statement: &a.statement,
            file: &a.file,
            parent: Some(a.parent),
            check: Some(&summary),
            siblings: &siblings,
            siblings_total: a.siblings.len(),
            branch_references: &a.branch_references,
            calls_left: 3,
            attempt: 1,
            file_name: "demo_thm.lean",
            services: &[],
            publish_plan_tool: None,
            read_plans_tool: None,
            exec_tool: None,
            bash_timeout_ms: None,
        };
        let system = default_template().render(&vars).unwrap();
        assert!(system.contains("errors: 15\n"), "{system}");
        assert!(system.contains("  - … and 3 more\n"), "{system}");
        assert!(system.contains("remaining sorries: 2\n"), "{system}");
        assert_eq!(system.matches("… and").count(), 1, "only the errors were cut: {system}");
        assert!(system.contains("from this same node (14);"), "{system}");
        assert!(system.contains("  (the 2 oldest are omitted)\n  - node 3: sorries"), "{system}");
        assert_eq!(system.matches("  - node ").count(), PROMPT_LIST_LIMIT);
    }

    #[test]
    fn an_invalid_verdict_shows_its_reason() {
        let mut c = check(CheckStatus::Invalid, &[], &[]);
        c.invalid_reason = Some("statement mismatch".into());
        let text = describe_check(&c);
        assert!(text.starts_with("status: invalid (statement mismatch)"), "{text}");
        assert!(check_feedback(&c, &text).contains("byte for byte"));
    }

    #[test]
    fn the_best_candidate_is_the_highest_status_and_the_latest_on_ties() {
        use CheckStatus::{Complete, Errors, Invalid, Sorries};
        let rounds = |s: &[Option<CheckStatus>]| -> Vec<(String, Option<CheckStatus>)> {
            s.iter().enumerate().map(|(i, s)| (format!("c{i}"), *s)).collect()
        };
        assert_eq!(pick_best(&rounds(&[])), None, "no candidate: submit nothing");
        assert_eq!(pick_best(&rounds(&[Some(Errors), Some(Sorries), Some(Errors)])), Some(1));
        assert_eq!(
            pick_best(&rounds(&[Some(Sorries), Some(Errors), Some(Sorries)])),
            Some(2),
            "ties → latest"
        );
        assert_eq!(pick_best(&rounds(&[Some(Invalid), Some(Complete), Some(Sorries)])), Some(1));
        assert_eq!(pick_best(&rounds(&[Some(Invalid), Some(Errors)])), Some(1));
        assert_eq!(pick_best(&rounds(&[Some(Invalid), Some(Invalid)])), Some(1));
        // A candidate the checker never judged loses to any judged one, but is
        // submitted when nothing was judged (the latest of them).
        assert_eq!(pick_best(&rounds(&[None, Some(Invalid), None])), Some(1));
        assert_eq!(pick_best(&rounds(&[None, None])), Some(1));
        assert!(status_rank(Complete) > status_rank(Sorries));
        assert!(status_rank(Sorries) > status_rank(Errors));
        assert!(status_rank(Errors) > status_rank(Invalid));
    }

    #[test]
    fn a_continuation_reply_without_a_fenced_block_is_a_failed_attempt() {
        assert_eq!(
            judge_continue_reply(&FinishReason::Stop, "here:\n```lean\nimport Mathlib\n```\nok", 5)
                .as_deref(),
            Ok("import Mathlib")
        );
        assert_eq!(
            judge_continue_reply(&FinishReason::Stop, "import Mathlib\ntheorem t : True := trivial", 5),
            Err(ReplyProblem::Empty),
            "prose without a block is not taken as the file"
        );
        assert_eq!(
            judge_continue_reply(&FinishReason::Length, "```lean\nimport Mathlib\n", 4096),
            Err(ReplyProblem::Truncated { output_tokens: 4096 })
        );
        assert_eq!(judge_continue_reply(&FinishReason::Stop, "```lean\n\n```", 2), Err(ReplyProblem::Empty));
    }

    /// The fixed loop against the `lean` service, played over the
    /// real framed channel: `check` carries the candidate alone
    /// (`CheckRequest` refuses any other field, `frozen` included) on the
    /// worker's own capability resource, where that service finds the run's
    /// statement.
    #[test]
    fn a_fixed_loop_check_carries_the_candidate_alone_on_the_capabilitys_resource() {
        use nucleus_protocol::{
            FromKernel, Payload, ProcessId, ResponseBody, ToKernel, read_frame, write_frame,
        };
        let proof = "import Mathlib\n\ntheorem demo_thm (n : Nat) : n + 0 = n := by simp";
        let (incoming, mut replies) = std::io::pipe().unwrap();
        let (calls, outgoing) = std::io::pipe().unwrap();
        let peer = std::thread::spawn(move || {
            write_frame(&mut replies, &FromKernel::Hello { process: ProcessId(7), handles: vec![] }).unwrap();
            let mut calls = std::io::BufReader::new(calls);
            let mut checked = Vec::new();
            while let Ok(ToKernel::Call { request, action, resource, payload, .. }) = read_frame(&mut calls) {
                let result = match action.as_str() {
                    model::ACTION_COMPLETE => Payload::encode(&model::CompletionResponse {
                        message: Message::text(Role::Assistant, format!("```lean\n{proof}\n```")),
                        finish_reason: FinishReason::Stop,
                        usage: model::Usage::default(),
                        served_by: "fixture".into(),
                    }),
                    lean::ACTION_CHECK => payload.decode::<lean::CheckRequest>().and_then(|request| {
                        checked.push((resource, request.candidate));
                        Payload::encode(&check(CheckStatus::Complete, &[], &[]))
                    }),
                    other => panic!("unexpected action {other}"),
                };
                let result = result.map(ResponseBody::Payload);
                write_frame(&mut replies, &FromKernel::Response { request, result }).unwrap();
            }
            checked
        });
        let options = Options::parse(["--model", "m", "--out", "/unused"].map(str::to_owned)).unwrap();
        let log =
            std::env::temp_dir().join(format!("nucleus-prover-fixed-check-{}.jsonl", std::process::id()));
        let own = ResourcePath::new(["deep", "demo_thm", "t7"]).unwrap();
        let mut worker = Worker {
            kernel: Connection::open(incoming, outgoing).unwrap(),
            options: &options,
            model: ModelAccess { handle: HandleId(1), pulls: None },
            model_resource: ResourcePath::new(["m"]).unwrap(),
            check: HandleId(2),
            check_resource: own.clone(),
            assignment: HandleId(3),
            submit: HandleId(4),
            task: own.clone(),
            attempts_log: std::fs::File::create(&log).unwrap(),
        };
        let end = worker.explore(&assignment()).unwrap();
        drop(worker);
        std::fs::remove_file(&log).unwrap();
        assert_eq!((end.stopped, end.calls), ("complete", 1), "the first candidate got its verdict");
        assert_eq!(peer.join().unwrap(), [(own, proof.to_owned())]);
    }
}
