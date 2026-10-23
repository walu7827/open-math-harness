"""Derive the local operator composition from the shipped VM composition.

The selected repository snapshot is deployment authority, just as deploy/vm's
JSON is. Its service graph, project identities, grants and lifecycle remain the
source of truth. This derivation removes the hosted platform/GitHub links,
its mirror-only scanner, and suggestions/Ultramath whose model calls require
output-token caps the CLI cannot enforce. They receive no local grants and
are not advertised as working services. It
selects the host completion broker and an owner-only Unix UI socket, and names
only the mathematical dependencies installed locally. It never enables a
native macOS fallback or a less privileged-looking substitute for enforcement.

Model grants are complete-only and scoped to codex. The model provider and
discovery retries are disabled. The CLI has no monetary charge report, so discovery explicitly has no
USD ceiling; the existing per-task model-call grants still bound generations.
The operator must read that limitation when starting a run. Image requests and
unsupported model parameters are refused by the broker.

No files or services are changed here. The root installer owns publication and
the launcher validates the resulting JSON before boot. A changed source shape
that no longer has a setting we replace raises an error, never silently drops
that local override. The resulting configuration must be kept away from the
projects user: changing it grants the authority of every booted service.
"""

import json
from pathlib import Path


MODEL = "codex/default"


def replace(args, old, new):
    offsets = [i for i in range(len(args) - len(old) + 1) if args[i:i + len(old)] == old]
    if len(offsets) != 1:
        raise ValueError("composition no longer has exactly one expected argument sequence: " + str(old))
    start = offsets[0]
    args[start:start + len(old)] = new


def make(source, lean_path, broker_port, *, memory_quiet_ms=300000):
    if type(broker_port) is not int or not 1 <= broker_port <= 65535:
        raise ValueError("invalid broker port")
    if type(memory_quiet_ms) is not int or memory_quiet_ms <= 0:
        raise ValueError("invalid memory quiet interval")
    if not lean_path or any(not value.startswith(("/opt/mathlib/", "/opt/lean/"))
                           or ".." in Path(value).parts for value in lean_path.split(":")):
        raise ValueError("Lean path must name only the published dependencies")
    config = json.loads((source / "deploy/vm/nucleus.json").read_text())
    programs = config["programs"]
    omitted = {"github", "platform", "swarm", "suggestions", "publish"}
    for name in ("github", "swarm", "suggestions", "publish"):
        del programs[name]
    config["boot"] = [entry for entry in config["boot"] if entry["program"] not in omitted]
    for entry in config["boot"]:
        grants = []
        for grant in entry["grants"]:
            if grant["endpoint"] in omitted:
                continue
            if grant["endpoint"] == "model":
                grant["actions"] = [action for action in grant["actions"] if action == "complete"]
                grant["resource"] = ["codex"]
                if not grant["actions"]:
                    continue
            grants.append(grant)
        entry["grants"] = grants
    for spec in config["parameters"].values():
        if spec["kind"] == "model":
            spec["default"] = MODEL
        elif spec["kind"] == "model_list":
            spec["default"] = [MODEL]
    config["parameters"]["model_base_url"]["default"] = f"http://127.0.0.1:{broker_port}"
    config["parameters"]["agent_model_deadline_ms"]["default"] = 180000
    config["parameters"]["discovery_reasoning_effort"]["default"] = "high"
    config["parameters"]["discovery_max_attempts"]["default"] = 16
    config["parameters"]["discovery_max_width"]["default"] = 2
    config["parameters"]["shell_max_vmem_mb"]["default"] = 16384
    programs["model"]["args"] += ["--retries", "0"]
    programs["model"]["binds"][0]["host"] = "/run/nucleus-local/model"
    programs["ui"] = {"executable": programs["ui"]["executable"], "args": ["--socket", "/ui/ui.sock"],
                      "binds": [{"host": "/run/nucleus-local/ui", "guest": "/ui", "writable": True}],
                      "network": False}
    memory = programs["memory"]["args"]
    replace(memory, ["--quiet-ms", "300000"], ["--quiet-ms", str(memory_quiet_ms)])
    # The full projects service still provides files, shell, PDF, skills and
    # jobs through the real projects identity. Its descriptions must not
    # advertise image-only tools or a mathematical environment not installed.
    projects = programs["projects"]["args"]
    shell_description = ("Commands run as the projects Linux user in this dedicated VM. All projects share that "
                         "user and home; the working directory is not an isolation boundary. The runtime's "
                         "files and credentials are inaccessible. Lean and public Mathlib are installed "
                         "read-only at /opt/lean and /opt/mathlib; use lean directly with LEAN_PATH. "
                         "Pass timeout_ms for long commands. Available additional tools depend on this VM.")
    for i, value in enumerate(projects):
        if isinstance(value, str) and value.startswith("LEAN_PATH="):
            projects[i] = "LEAN_PATH=" + lean_path
        elif isinstance(value, str) and value.startswith("PATH=/opt/lean/bin:"):
            projects[i] = "PATH=/opt/lean/bin:/usr/local/bin:/usr/bin:/bin"
        elif isinstance(value, str) and value.startswith("This is the VM itself, not a sandbox:"):
            projects[i] = shell_description
    lean = programs["lean"]
    index = lean["args"].index("--lean-path") + 1
    lean["args"][index] = lean_path
    lean["binds"] = [bind for bind in lean["binds"] if bind["host"] != "/opt/cslib"]
    discovery = programs["discovery"]["args"]
    replace(discovery, ["--prover-arg", "--temperature", "--prover-arg", "1"], [])
    replace(discovery, ["--max-usd-per-problem", {"param": "discovery_max_usd_per_problem"}],
            ["--no-usd-ceiling"])
    replace(discovery, ["--prover-arg", "--model-deadline-ms", "--prover-arg", "3600000"],
            ["--prover-arg", "--model-deadline-ms", "--prover-arg", "180000"])
    discovery += ["--provider-retries", "0", "--prover-arg", "--model-retries", "--prover-arg", "0"]
    parent = programs["parent"]["args"]
    replace(parent, ["--project-scope", "github"], [])
    replace(parent, ["--session-scope", "swarm"], [])
    replace(parent, ["--child-arg", "--organisation"], [])
    replace(parent, ["--child-arg", "--image-model", "--child-arg", {"param": "image_model"}], [])
    parent += ["--child-arg", "--model", "--child-arg", MODEL,
               "--child-arg", "--model-retries", "--child-arg", "0"]
    index = parent.index("--tool-bindings") + 1
    parent[index] = json.dumps([binding for binding in json.loads(parent[index])
                               if binding["endpoint"] not in ("github", "platform")])
    # Remove parameters whose arguments were deliberately removed above.
    used = set()
    for program in programs.values():
        for arg in program["args"]:
            if isinstance(arg, dict):
                used.add(arg.get("param", arg.get("each")))
    config["parameters"] = {name: spec for name, spec in config["parameters"].items() if name in used}
    return config
