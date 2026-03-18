#!/usr/bin/env python3
"""BioAgentics dispatcher — owns agent lifecycle, presence, and scheduling."""

import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from bioagentics.config import AgentConfig, DispatchConfig, api, load_config, API_URL, REPO_ROOT

PID_FILE = REPO_ROOT / ".dispatch.pid"

# Roles that generate work and should always run (no task check)
GENERATORS = {
    "RESEARCH_DIRECTOR",
    "LITERATURE_REVIEWER",
    "DATA_CURATOR",
    "SYSTEMS_ENGINEER",
}

# Retry strategies by exit code (None = don't retry)
NO_RETRY_CODES = {0, 2}

# Module-level dispatch config, set in main()
_dispatch: DispatchConfig = DispatchConfig()

# Shutdown coordination
_stop_event = threading.Event()
_children: list[subprocess.Popen] = []
_children_lock = threading.Lock()
_git_lock = threading.Lock()


@dataclass
class RunStats:
    """Stats captured from the claude stream-json result event."""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: float = 0.0
    num_turns: int = 0
    session_id: str = ""


def _is_stopping() -> bool:
    return _stop_event.is_set()


def _interruptible_sleep(seconds: int):
    """Sleep that returns immediately when stop is requested."""
    _stop_event.wait(timeout=seconds)


def _terminate_children():
    """Terminate all running child processes."""
    with _children_lock:
        for proc in _children:
            try:
                proc.terminate()
            except OSError:
                pass


def journal(content: str, project: str | None = None, division: str | None = None):
    """Write a dispatcher journal entry."""
    payload: dict = {"username": "dispatcher", "content": content}
    if division:
        payload["division"] = division
    if project:
        payload["project"] = project
    api("POST", "/journal", json=payload)


def set_presence(username: str, status: str, project: str | None = None, division: str | None = None):
    payload: dict = {"username": username, "status": status}
    if division:
        payload["division"] = division
    if project:
        payload["project"] = project
    api("POST", "/agents", json=payload)


def clear_all_presence():
    """Reset all agents to idle on startup."""
    resp = api("GET", "/agents?status=running")
    if resp and resp.ok:
        stuck = resp.json().get("items", [])
        if stuck:
            for agent in stuck:
                username = agent["username"]
                division = agent.get("division") or None
                project = agent.get("project") or None
                label = f"{username}/{project}" if project else username
                print(f"  cleanup: {label} was stuck as 'running', setting idle")
                set_presence(username, "idle", project, division)
            journal(
                f"startup cleanup: reset {len(stuck)} stuck agent(s) to idle: "
                + ", ".join(
                    f"{a['username']}/{a.get('project', '')}" if a.get("project") else a["username"]
                    for a in stuck
                )
            )


def reset_stuck_tasks():
    """Reset in_progress tasks back to pending on startup — no agents are running yet."""
    resp = api("GET", "/tasks?status=in_progress&limit=100")
    if not resp or not resp.ok:
        return
    stuck = resp.json().get("items", [])
    if not stuck:
        return
    for task in stuck:
        api("PATCH", f"/tasks/{task['id']}", json={"status": "pending"})
    labels = ", ".join(f"#{t['id']} {t['title']}" for t in stuck)
    print(f"  cleanup: reset {len(stuck)} stuck in_progress task(s) to pending")
    journal(f"startup cleanup: reset {len(stuck)} stuck in_progress task(s) to pending: {labels}")


def get_work(username: str, division: str | None = None) -> list[str] | None:
    """Return projects with pending/in_progress work, or None if no work at all.

    Returns sorted project list (may be empty if tasks have no project field).
    Returns None when no pending/in_progress tasks exist for this agent.
    """
    projects: set[str] = set()
    has_any = False
    div_filter = f"&division={division}" if division else ""
    for status in ("pending", "in_progress"):
        resp = api("GET", f"/tasks?username={username}&status={status}&limit=100{div_filter}")
        if resp and resp.ok:
            for task in resp.json().get("items", []):
                has_any = True
                proj = task.get("project")
                if proj:
                    projects.add(proj)
    if not has_any:
        return None
    return sorted(projects)


def project_locked_by(project: str, division: str | None = None) -> str | None:
    """Return the role running on this project, or None if unlocked."""
    resp = api("GET", "/agents?status=running")
    if resp and resp.ok:
        for agent in resp.json().get("items", []):
            if agent.get("project") == project:
                if division and agent.get("division") != division:
                    continue
                return agent["username"]
    return None


def clear_stale_presences():
    """Reset agents stuck as 'running' beyond the presence timeout.

    Catches the case where set_presence("idle") failed in run_agent's
    finally block (e.g. API timeout), leaving an agent permanently stuck
    and blocking all future dispatches for its project.
    """
    resp = api("GET", "/agents?status=running")
    if not resp or not resp.ok:
        return
    now = datetime.now(timezone.utc)
    for agent in resp.json().get("items", []):
        updated = agent.get("updated_at", "")
        if not updated:
            continue
        try:
            last_seen = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            continue
        age = (now - last_seen).total_seconds()
        if age > _dispatch.presence_timeout:
            username = agent["username"]
            division = agent.get("division") or None
            project = agent.get("project") or None
            label = f"{username}/{project}" if project else username
            print(f"  STALE: {label} running for {int(age)}s — resetting to idle")
            set_presence(username, "idle", project, division)
            journal(
                f"stale presence cleanup: {label} was running for "
                f"{int(age)}s (timeout {_dispatch.presence_timeout}s), reset to idle",
                project,
                division,
            )


def check_stale_blocked_tasks():
    """Log warnings for tasks blocked longer than 3 hours."""
    resp = api("GET", "/tasks?status=blocked&older_than=3h&limit=100")
    if resp and resp.ok:
        total = resp.json().get("total", 0)
        if total > 0:
            print(f"  WARNING: {total} task(s) blocked >3h")
            lines = []
            for task in resp.json().get("items", []):
                lines.append(
                    f"- [{task['id']}] {task['title']} (assigned: {task['username']})"
                )
                print(f"    {lines[-1]}")
            journal(f"STALE BLOCKED: {total} task(s) blocked >3h:\n" + "\n".join(lines))


def _project_is_idle(name: str, division: str | None = None) -> bool:
    """True if project has no active/blocked tasks but at least one done task."""
    div_filter = f"&division={division}" if division else ""
    for status in ("pending", "in_progress", "blocked"):
        task_resp = api("GET", f"/tasks?project={name}&status={status}&limit=1{div_filter}")
        if task_resp and task_resp.ok and task_resp.json().get("total", 0) > 0:
            return False
    done_resp = api("GET", f"/tasks?project={name}&status=done&limit=1{div_filter}")
    return (
        done_resp is not None
        and done_resp.ok
        and done_resp.json().get("total", 0) > 0
    )


def _advance_projects(from_status: str, to_status: str):
    """Advance projects from one status to the next when all tasks are done."""
    resp = api("GET", f"/projects?status={from_status}&limit=100")
    if not resp or not resp.ok:
        return

    for proj in resp.json().get("items", []):
        name = proj["name"]
        div = proj.get("division")
        if not _project_is_idle(name, div):
            continue

        patch_resp = api("PATCH", f"/projects/{name}", json={"status": to_status})
        if patch_resp and patch_resp.ok:
            print(f"  ADVANCE: {name} {from_status} → {to_status} (no active tasks)")
            journal(
                f"auto-advanced from {from_status} to {to_status} — "
                "no pending/in_progress tasks remain",
                name,
                div,
            )
        else:
            print(f"  WARNING: failed to advance {name} to {to_status}")


def _project_has_no_tasks(name: str, division: str | None = None) -> bool:
    """True if project has zero tasks in any status."""
    div_filter = f"&division={division}" if division else ""
    for status in ("pending", "in_progress", "blocked", "done", "cancelled"):
        resp = api("GET", f"/tasks?project={name}&status={status}&limit=1{div_filter}")
        if resp and resp.ok and resp.json().get("total", 0) > 0:
            return False
    return True


def _has_active_task_for(username: str, project: str, division: str | None = None) -> bool:
    """True if an active (pending/in_progress/blocked) task exists for this agent+project."""
    div_filter = f"&division={division}" if division else ""
    for st in ("pending", "in_progress", "blocked"):
        resp = api(
            "GET",
            f"/tasks?username={username}&project={project}&status={st}&limit=1{div_filter}",
        )
        if resp and resp.ok and resp.json().get("total", 0) > 0:
            return True
    return False


def check_planning_projects():
    """Create project_manager tasks for projects stuck in planning with no PM task."""
    resp = api("GET", "/projects?status=planning&limit=100")
    if not resp or not resp.ok:
        return

    for proj in resp.json().get("items", []):
        name = proj["name"]
        div = proj.get("division") or "cancer"
        if _has_active_task_for("project_manager", name, div):
            continue

        print(f"  STUCK PLANNING: {name} — creating project_manager task")
        api(
            "POST",
            "/tasks",
            json={
                "username": "project_manager",
                "title": f"Create development tasks for {name}",
                "description": (
                    f"Project {name} is in planning but has no project_manager task.\n\n"
                    f"Read the research plan at plans/{div}/{name}.md (or via get_project()) "
                    f"and create developer tasks to implement it."
                ),
                "project": name,
                "division": div,
                "priority": 3,
            },
        )
        journal(
            f"auto-created project_manager task — project stuck in planning with no PM task",
            name,
            div,
        )


# Expected agent role for each pipeline stage
_STAGE_AGENTS = {
    "development": "developer",
    "analysis": "analyst",
    "validation": "validation_scientist",
    "documentation": "research_writer",
}


def check_orphaned_pipeline_projects():
    """Warn about projects in an active pipeline stage with zero tasks."""
    for stage, expected_agent in _STAGE_AGENTS.items():
        resp = api("GET", f"/projects?status={stage}&limit=100")
        if not resp or not resp.ok:
            continue
        for proj in resp.json().get("items", []):
            name = proj["name"]
            div = proj.get("division")
            if not _project_has_no_tasks(name, div) or _project_is_idle(name, div):
                continue
            # Project is in an active stage but has no tasks at all
            print(f"  ORPHANED: {name} in '{stage}' with no tasks")
            journal(
                f"orphaned project: status is '{stage}' but no tasks exist — "
                f"expected {expected_agent} tasks. May need project_manager intervention.",
                name,
                div,
            )


def check_stale_pipeline_projects():
    """Advance projects through the research pipeline when their tasks complete.

    Pipeline: development → analysis → validation → documentation → published
    """
    _advance_projects("development", "analysis")
    _advance_projects("analysis", "validation")
    _advance_projects("validation", "documentation")
    _advance_projects("documentation", "published")




def check_published_missing_reports():
    """Create research_writer tasks for published projects missing report files."""
    resp = api("GET", "/projects?status=published&limit=100")
    if not resp or not resp.ok:
        return

    for proj in resp.json().get("items", []):
        name = proj["name"]
        div = proj.get("division") or "cancer"
        report_path = Path(f"reports/{div}/{name}.md")
        if report_path.exists():
            continue

        # Check if a task already exists for this (pending, in_progress, or blocked)
        has_existing = False
        for st in ("pending", "in_progress", "blocked"):
            task_resp = api(
                "GET",
                f"/tasks?username=research_writer&project={name}&division={div}"
                f"&status={st}&limit=1",
            )
            if task_resp and task_resp.ok and task_resp.json().get("total", 0) > 0:
                has_existing = True
                break
        if has_existing:
            continue

        # Create the task
        print(f"  REPORT MISSING: {div}/{name} — creating research_writer task")
        api(
            "POST",
            "/tasks",
            json={
                "username": "research_writer",
                "title": f"Write research report for {name}",
                "description": (
                    f"Project {name} is published but missing its research report.\n\n"
                    f"Write a comprehensive report at reports/{div}/{name}.md covering:\n"
                    f"- Executive summary, background, methodology, results, "
                    f"discussion, limitations, next steps, references.\n\n"
                    f"Review the research plan, journal entries from analyst and "
                    f"validation_scientist, and code/results in data/results/{name}/.\n\n"
                    f"Also update the project via update_project() with:\n"
                    f"- findings_content: the technical findings summary\n"
                    f"- plain_summary: a 2-4 sentence plain English overview for "
                    f"non-scientists explaining what was studied, what was found, "
                    f"and why it matters — no jargon\n"
                    f"- impact_score: one of breakthrough/high/moderate/incremental"
                ),
                "project": name,
                "division": div,
                "priority": 3,
            },
        )
        journal(
            f"auto-created research_writer task — reports/{div}/{name}.md missing "
            f"for published project",
            name,
            div,
        )


def validate_summary(role: str, project: str | None = None, division: str | None = None):
    """Check if the summary file has a YAML frontmatter header."""
    div = f".{division}" if division else ""
    if project:
        path = Path(f"cache/{role}{div}.{project}.summary")
    else:
        path = Path(f"cache/{role}{div}.summary")

    if not path.exists():
        return

    content = path.read_text()
    if not content.startswith("---"):
        print(f"  warning: {path} missing YAML frontmatter header")


def log_run(
    agent: str,
    backend: str,
    model: str,
    project: str | None,
    started_at: str,
    finished_at: str,
    exit_code: int,
    duration_seconds: int,
    stats: RunStats | None = None,
    division: str | None = None,
):
    """Log a run to the agent-comms API."""
    payload: dict = {
        "agent": agent,
        "backend": backend,
        "model": model,
        "started_at": started_at,
        "finished_at": finished_at,
        "exit_code": exit_code,
        "duration_seconds": duration_seconds,
    }
    if division:
        payload["division"] = division
    if project:
        payload["project"] = project
    if stats:
        payload["input_tokens"] = stats.input_tokens
        payload["output_tokens"] = stats.output_tokens
        payload["cache_read_tokens"] = stats.cache_read_tokens
        payload["cache_creation_tokens"] = stats.cache_creation_tokens
        payload["cost_usd"] = stats.cost_usd
    api("POST", "/runs", json=payload)


def get_retry_strategy(exit_code: int) -> dict | None:
    """Get retry strategy for a given exit code."""
    if exit_code in NO_RETRY_CODES:
        return None
    if exit_code in (137, 139):
        return {"delays": _dispatch.oom_delays, "max_attempts": _dispatch.oom_max_attempts}
    return {"delays": _dispatch.retry_delays, "max_attempts": _dispatch.retry_max_attempts}


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_agent_command(config: AgentConfig, project: str | None, division: str | None = None) -> list[str]:
    """Build the CLI command to invoke an agent."""
    role = config.role
    div = division or config.division

    # Use project-specific summary when scoped to a project, global otherwise
    if project:
        summary_file = f"cache/{role}.{div}.{project}.summary"
    else:
        summary_file = f"cache/{role}.{div}.summary"

    division_clause = (
        f"Division: You are in the '{div}' division. "
        f"Always pass division=\"{div}\" when creating or querying "
        f"tasks, projects, and journal entries."
    )

    project_clause = ""
    if project:
        project_clause = (
            f"Focus exclusively on project: {project}. "
            f"Only work on tasks where project={project}. "
            f"All work happens in this single repository."
        )

    if config.backend == "claude":
        prompt = (
            f"0. Read @org-roles/{div}/{role}.md .\n"
            f"  1. Read previous context summary at @{summary_file} (if exists).\n"
            f"  2. {division_clause} {project_clause}\n"
            f"  3. Do your job. Note: your presence (running/idle) is managed by "
            f"the dispatcher — do not manage your own agent presence.\n"
            f"  4. Finally, save a new short and concise context summary to "
            f"{summary_file} for next run. Use the format in @SUMMARY_FORMAT.md ."
        )
        return [
            "claude",
            "-p",
            prompt,
            "--dangerously-skip-permissions",
            "--no-session-persistence",
            "--output-format",
            "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--model",
            config.model,
            "--effort",
            config.effort,
        ]

    elif config.backend == "codex":
        role_content = Path(f"org-roles/{div}/{role}.md").read_text()
        try:
            summary = Path(summary_file).read_text()
        except FileNotFoundError:
            summary = "(no prior context)"

        prompt = (
            f"You are: {role}\n\n"
            f"{role_content}\n\n"
            f"Previous context summary:\n{summary}\n\n"
            f"{division_clause}\n\n"
            f"{project_clause}\n\n"
            f"Do your job. Your presence (running/idle) is managed by the "
            f"dispatcher — do not manage your own agent presence.\n\n"
            f"When done, save a short context summary to {summary_file}."
        )
        return [
            "codex",
            "--model",
            config.model,
            "--full-auto",
            "--prompt",
            prompt,
        ]

    else:
        raise ValueError(f"unknown backend: {config.backend}")


def run_agent(config: AgentConfig, project: str | None = None) -> int:
    """Invoke an agent with retry logic, managing its presence lifecycle."""
    division = config.division
    label = f"{config.role}" + (f"/{project}" if project else "")
    username = config.role.lower()

    attempt = 0

    while not _is_stopping():
        attempt += 1

        # --- Dispatcher owns presence ---
        set_presence(username, "running", project, division)

        cmd = build_agent_command(config, project, division)
        print(
            f"  dispatch: role={config.role} backend={config.backend} "
            f"model={config.model} effort={config.effort} "
            f"project={project or '<all>'}"
        )

        started_at = utcnow()
        start_time = time.monotonic()
        print(f"  START: {label} (attempt {attempt})")

        # Capture stdout to parse stream-json result event for stats
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        with _children_lock:
            _children.append(proc)

        stats: RunStats | None = None
        try:
            # Stream stdout line by line, tee to terminal, parse result event
            if proc.stdout is None:
                raise RuntimeError(f"Failed to capture stdout for {label}")
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") == "result":
                        usage = event.get("usage", {})
                        stats = RunStats(
                            input_tokens=usage.get("input_tokens", 0),
                            output_tokens=usage.get("output_tokens", 0),
                            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
                            cache_creation_tokens=usage.get("cache_creation_input_tokens", 0),
                            cost_usd=event.get("total_cost_usd", 0.0) or 0.0,
                            num_turns=event.get("num_turns", 0),
                            session_id=event.get("session_id", ""),
                        )
                except (json.JSONDecodeError, TypeError):
                    pass
            proc.wait()
            exit_code = proc.returncode
        except Exception as e:
            print(f"  ERROR: {label} — {e}")
            exit_code = 1
        finally:
            with _children_lock:
                if proc in _children:
                    _children.remove(proc)
            # --- Always clean up presence (must include division+project for composite PK) ---
            set_presence(username, "idle", project, division)

        # If we're stopping, don't log or retry — just exit
        if _is_stopping():
            return exit_code

        finished_at = utcnow()
        duration = int(time.monotonic() - start_time)

        if stats:
            print(
                f"  STATS: {label} — "
                f"in={stats.input_tokens} out={stats.output_tokens} "
                f"cache_read={stats.cache_read_tokens} "
                f"cost=${stats.cost_usd:.4f} turns={stats.num_turns}"
            )

        # Log run to API
        log_run(
            agent=config.role.lower(),
            backend=config.backend,
            model=config.model,
            project=project,
            started_at=started_at,
            finished_at=finished_at,
            exit_code=exit_code,
            duration_seconds=duration,
            stats=stats,
            division=division,
        )

        if exit_code == 0:
            print(f"  DONE: {label} ({duration}s)")
            journal(f"completed {label} in {duration}s", project, division)
            # Validate summary format
            validate_summary(config.role, project, division)
            # Commit cache summary + research artifacts (lock prevents races)
            with _git_lock:
                # 1. Commit cache summaries
                cache_dir = REPO_ROOT / "cache"
                subprocess.run(
                    ["git", "add"] + [str(p) for p in cache_dir.glob("*summary")],
                    capture_output=True, cwd=REPO_ROOT,
                )
                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        f"run_agent: commit {config.role} summary cache",
                    ],
                    capture_output=True, cwd=REPO_ROOT,
                )

                # 2. Commit plans, reports, output, and project-specific source files
                artifact_paths: list[str] = []
                plans_dir = REPO_ROOT / "plans"
                if plans_dir.is_dir():
                    for p in plans_dir.rglob("*.md"):
                        artifact_paths.append(str(p))
                reports_dir = REPO_ROOT / "reports"
                if reports_dir.is_dir():
                    for p in reports_dir.rglob("*.md"):
                        artifact_paths.append(str(p))
                output_dir = REPO_ROOT / "output"
                if output_dir.is_dir():
                    for p in output_dir.rglob("*"):
                        if p.is_file() and p.name != ".DS_Store":
                            artifact_paths.append(str(p))
                # Project-specific pipeline scripts (not the core bioagentics package)
                src_dir = REPO_ROOT / "src"
                if src_dir.is_dir():
                    for child in src_dir.iterdir():
                        if child.is_dir() and child.name != "bioagentics":
                            for p in child.rglob("*.py"):
                                artifact_paths.append(str(p))
                pipelines_dir = REPO_ROOT / "pipelines"
                if pipelines_dir.is_dir():
                    for p in pipelines_dir.rglob("*.py"):
                        artifact_paths.append(str(p))
                if artifact_paths:
                    subprocess.run(
                        ["git", "add", "--"] + artifact_paths,
                        capture_output=True, cwd=REPO_ROOT,
                    )
                    # Only commit if there are staged changes
                    diff = subprocess.run(
                        ["git", "diff", "--cached", "--quiet"],
                        capture_output=True, cwd=REPO_ROOT,
                    )
                    if diff.returncode != 0:
                        artifact_msg = f"run_agent: commit research artifacts after {config.role}"
                        if project:
                            artifact_msg += f" [{project}]"
                        subprocess.run(
                            ["git", "commit", "-m", artifact_msg],
                            capture_output=True, cwd=REPO_ROOT,
                        )

                push = subprocess.run(
                    ["git", "push"], capture_output=True, cwd=REPO_ROOT,
                )
                if push.returncode != 0:
                    print(
                        f"  warning: git push failed for {config.role} "
                        f"(exit {push.returncode})"
                    )
            return 0

        # Determine retry strategy
        strategy = get_retry_strategy(exit_code)

        if strategy is None:
            if exit_code == 2:
                msg = f"skipped {label} — bad input/config (exit 2), not retrying"
                print(f"  SKIP: {label} — bad input/config (exit 2), not retrying")
            else:
                msg = f"finished {label} (exit {exit_code})"
                print(f"  DONE: {label} (exit {exit_code})")
            journal(msg, project, division)
            return exit_code

        if attempt >= strategy["max_attempts"]:
            print(
                f"  GIVE UP: {label} — failed {attempt} attempts (last exit {exit_code})"
            )
            journal(
                f"gave up on {label} after {attempt} attempts (exit {exit_code})",
                project,
                division,
            )
            return exit_code

        delay = strategy["delays"][min(attempt - 1, len(strategy["delays"]) - 1)]
        print(
            f"  RETRY: {label} — exit {exit_code}, waiting {delay}s (attempt {attempt}/{strategy['max_attempts']})"
        )
        _interruptible_sleep(delay)

    return 1  # stopped


def dispatch_cycle(agents: list[AgentConfig], allow_research_director: bool):
    """Run one dispatch cycle.

    Generators and task-driven agents run in parallel.  Generators create
    work (research proposals, literature scans, etc.) while task-driven
    agents process existing work items concurrently.
    """
    if _is_stopping():
        return

    # Clear agents stuck as 'running' beyond the presence timeout
    clear_stale_presences()

    # Check for stale blocked tasks
    check_stale_blocked_tasks()

    # Advance projects through the research pipeline
    check_stale_pipeline_projects()

    # Recover stuck projects
    check_planning_projects()
    check_orphaned_pipeline_projects()

    # Create tasks for published projects missing research reports
    check_published_missing_reports()

    # Build the full work queue: generators + task-driven agents
    generator_queue: list[tuple[AgentConfig, str | None]] = []
    task_queue: list[tuple[AgentConfig, str | None]] = []

    # Collect generators (unscoped — they decide their own work)
    for config in agents:
        if config.role not in GENERATORS:
            continue
        if config.role == "RESEARCH_DIRECTOR" and not allow_research_director:
            print(f"  SKIP: {config.role}/{config.division} (disabled)")
            continue
        generator_queue.append((config, None))

    # Collect task-driven agents (scoped to projects with pending work)
    for config in agents:
        if config.role in GENERATORS:
            continue

        username = config.role.lower()
        projects = get_work(username, config.division)
        if projects is None:
            print(f"  SKIP: {config.role}/{config.division} (no pending tasks)")
            continue

        if projects:
            for proj in projects:
                task_queue.append((config, proj))
        else:
            # Tasks exist but no project field — run unscoped
            task_queue.append((config, None))

    # Interleave: task-driven agents first (they do concrete work),
    # then generators (they create new work for future cycles)
    work_queue: list[tuple[AgentConfig, str | None]] = task_queue + generator_queue

    if not work_queue:
        print("  no work queued")
        return

    # Execute all agents in parallel (respecting project locks)
    print(f"  dispatching {len(work_queue)} agent/project pair(s)...")

    # Track (role, project, division) triples claimed this cycle to prevent
    # the same agent from running twice on the same project.  Different
    # agents CAN work on the same project concurrently — they operate on
    # separate tasks and coordinate through the API.
    claimed: set[tuple[str, str, str | None]] = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=_dispatch.max_workers) as executor:
        futures: dict[concurrent.futures.Future, str] = {}

        for config, proj in work_queue:
            if _is_stopping():
                break
            label = f"{config.role}" + (f"/{proj}" if proj else "")
            role = config.role.lower()

            if proj:
                # Same role+division already claimed this project this cycle
                claim_key = (role, proj, config.division)
                if claim_key in claimed:
                    print(f"  SKIP: {label} (already dispatched)")
                    continue
                # Same role already running on this project from a prior cycle
                locked_by = project_locked_by(proj, config.division)
                if locked_by == role:
                    print(f"  SKIP: {label} (already running)")
                    journal(f"skipped {label} — already running on this project", proj, config.division)
                    continue
                claimed.add(claim_key)

            future = executor.submit(run_agent, config, proj)
            futures[future] = label

        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"  EXCEPTION: {label} — {e}")


def _read_pid() -> int | None:
    """Read PID from file, return None if stale or missing."""
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # check if process exists
        return pid
    except (ValueError, ProcessLookupError, PermissionError):
        PID_FILE.unlink(missing_ok=True)
        return None


def _write_pid():
    PID_FILE.write_text(str(os.getpid()))


def _remove_pid():
    PID_FILE.unlink(missing_ok=True)


def stop():
    """Stop a running dispatcher."""
    pid = _read_pid()
    if pid is None:
        print("no dispatcher running")
        return
    print(f"stopping dispatcher (pid {pid})...")
    os.kill(pid, signal.SIGTERM)
    # Wait for it to exit
    for _ in range(60):
        try:
            os.kill(pid, 0)
            time.sleep(0.5)
        except ProcessLookupError:
            print("dispatcher stopped")
            _remove_pid()
            return
    print(f"dispatcher (pid {pid}) did not exit after 30s — sending SIGKILL...")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    _remove_pid()
    print("dispatcher killed")


def status():
    """Check if the dispatcher is running."""
    pid = _read_pid()
    if pid is None:
        print("dispatcher is not running")
    else:
        print(f"dispatcher is running (pid {pid})")


_shutdown_once = threading.Event()


def _shutdown():
    """Clean shutdown: signal stop, terminate children, clear presence, remove PID."""
    if _shutdown_once.is_set():
        return
    _shutdown_once.set()
    _stop_event.set()
    _terminate_children()
    # Wait briefly for children to exit
    with _children_lock:
        procs = list(_children)
    for proc in procs:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
    clear_all_presence()
    _remove_pid()


def main():
    global _dispatch

    # Handle subcommands
    if len(sys.argv) > 1 and sys.argv[1] == "stop":
        stop()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
        return

    # Check for already-running dispatcher
    existing = _read_pid()
    if existing is not None:
        print(f"error: dispatcher already running (pid {existing})")
        print("  use 'bioagentics-dispatch stop' to stop it first")
        sys.exit(1)

    allow_research_director = len(sys.argv) > 1 and sys.argv[1] == "yes"
    agents, _dispatch = load_config()

    print(
        f"bioagentics dispatcher starting (research_director={'enabled' if allow_research_director else 'disabled'})"
    )
    print(f"  api: {API_URL}")
    print(f"  agents: {', '.join(a.role for a in agents)}")
    print(
        f"  timing: short={_dispatch.short_break}s long={_dispatch.long_break}s "
        f"every={_dispatch.long_break_every} workers={_dispatch.max_workers}"
    )

    # Write PID file
    _write_pid()

    # Clean up stale state from previous crashed run
    print("startup: clearing stale agent presence...")
    clear_all_presence()
    print("startup: checking for stuck in_progress tasks...")
    reset_stuck_tasks()

    # Signal handler only sets the stop event — actual cleanup happens in the
    # finally block via _shutdown().  Calling _shutdown() here would risk
    # deadlocking on _children_lock or hanging on API calls.
    def handle_signal(sig, _frame):
        signame = signal.Signals(sig).name
        print(f"\n{signame} received — shutting down...")
        _stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        iteration = 0
        while not _is_stopping():
            print(f"\n{'=' * 40}")
            print(f"iteration {iteration}")
            print(f"{'=' * 40}")

            dispatch_cycle(agents, allow_research_director)

            if _is_stopping():
                break

            if iteration > 0 and iteration % _dispatch.long_break_every == 0:
                print(f"long break: {_dispatch.long_break}s...")
                _interruptible_sleep(_dispatch.long_break)
            else:
                print(f"short break: {_dispatch.short_break}s...")
                _interruptible_sleep(_dispatch.short_break)

            iteration += 1
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
