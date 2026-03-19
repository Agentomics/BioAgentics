#!/usr/bin/env python3
"""BioAgentics MCP server — exposes agent-comms API and research tools."""

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import quote, urlencode

import requests
from mcp.server.fastmcp import FastMCP

from bioagentics.config import (
    API_PREFIX,
    API_URL,
    DATA_DIR,
    HEADERS,
    REPO_ROOT,
)

GITHUB_ORG = os.environ.get("GITHUB_ORG", "bioagentics")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "BioAgentics")

mcp = FastMCP("bioagentics")


def _api(method: str, path: str, **kwargs) -> dict:
    """Make an API request and return the JSON response."""
    try:
        resp = requests.request(
            method, f"{API_URL}{API_PREFIX}{path}", headers=HEADERS, timeout=30, **kwargs
        )
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        return {"error": str(e)}


def _gh(*args: str, cwd: str | Path | None = None) -> str:
    """Run a gh CLI command and return output."""
    try:
        result = subprocess.run(
            ["gh", *args],
            capture_output=True,
            text=True,
            cwd=cwd or REPO_ROOT,
            timeout=60,
        )
    except FileNotFoundError:
        return "Error: gh CLI not installed (https://cli.github.com)"
    except subprocess.TimeoutExpired:
        return "Error: gh command timed out after 60s"
    output = result.stdout.strip()
    if result.returncode != 0:
        err = result.stderr.strip()
        if err:
            output = f"{output}\n{err}".strip() if output else err
        output += f"\n[exit code: {result.returncode}]"
    return output


# ── Task Tools ──


@mcp.tool()
def list_tasks(
    username: str = "",
    status: str = "",
    division: str = "",
    project: str = "",
    priority: int = 0,
    search: str = "",
    sort: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List tasks from the agent coordination API.

    Filter by username, status (pending/in_progress/blocked/done/cancelled),
    division, project, priority (1-5), or search text (matches title/description).
    Sort: desc (newest first, default) or asc.
    Returns highest priority first within sort order.
    """
    qp: dict[str, str | int] = {"limit": limit, "offset": offset, "sort": sort}
    if username:
        qp["username"] = username
    if status:
        qp["status"] = status
    if division:
        qp["division"] = division
    if project:
        qp["project"] = project
    if priority:
        qp["priority"] = priority
    if search:
        qp["search"] = search
    return json.dumps(_api("GET", f"/tasks?{urlencode(qp)}"), indent=2)


@mcp.tool()
def get_task(task_id: int) -> str:
    """Get a specific task by ID."""
    return json.dumps(_api("GET", f"/tasks/{task_id}"), indent=2)


@mcp.tool()
def create_task(
    username: str,
    title: str,
    status: str = "pending",
    division: str = "",
    project: str = "",
    description: str = "",
    priority: int = 3,
) -> str:
    """Create a new task assigned to an agent.

    Status: pending, in_progress, blocked, done, cancelled.
    Priority: 1 (lowest) to 5 (highest).
    Division: research division (e.g. cancer, crohns).
    """
    payload: dict = {
        "username": username,
        "title": title,
        "status": status,
        "priority": priority,
    }
    if division:
        payload["division"] = division
    if project:
        payload["project"] = project
    if description:
        payload["description"] = description
    return json.dumps(_api("POST", "/tasks", json=payload), indent=2)


@mcp.tool()
def update_task(
    task_id: int,
    status: str = "",
    title: str = "",
    description: str = "",
    priority: int = 0,
    blocked_reason: str = "",
) -> str:
    """Update a task's status and/or details.

    Status lifecycle: pending -> in_progress -> done (or in_progress -> blocked -> in_progress -> done).
    When blocking, include blocked_reason. The API auto-sets blocked_at timestamps.
    Priority: 1 (lowest) to 5 (highest). Pass 0 to leave unchanged.
    """
    payload: dict = {}
    if status:
        payload["status"] = status
    if title:
        payload["title"] = title
    if description:
        payload["description"] = description
    if priority:
        payload["priority"] = priority
    if blocked_reason:
        payload["blocked_reason"] = blocked_reason
    if not payload:
        return json.dumps({"error": "no fields to update"})
    return json.dumps(_api("PATCH", f"/tasks/{task_id}", json=payload), indent=2)


# ── Journal Tools ──


@mcp.tool()
def list_journal(
    username: str = "",
    division: str = "",
    project: str = "",
    search: str = "",
    sort: str = "desc",
    limit: int = 50,
    offset: int = 0,
) -> str:
    """List journal entries (shared memory between agents).

    Filter by username, division, project, or search text (matches content).
    Sort: desc (newest first, default) or asc.
    """
    qp: dict[str, str | int] = {"limit": limit, "offset": offset, "sort": sort}
    if username:
        qp["username"] = username
    if division:
        qp["division"] = division
    if project:
        qp["project"] = project
    if search:
        qp["search"] = search
    return json.dumps(_api("GET", f"/journal?{urlencode(qp)}"), indent=2)


@mcp.tool()
def create_journal(username: str, content: str, division: str = "", project: str = "") -> str:
    """Write a journal entry. The journal is shared memory between all agents."""
    payload: dict = {"username": username, "content": content}
    if division:
        payload["division"] = division
    if project:
        payload["project"] = project
    return json.dumps(_api("POST", "/journal", json=payload), indent=2)


# ── Project Tools ──


@mcp.tool()
def list_projects(status: str = "", division: str = "", labels: str = "", limit: int = 100) -> str:
    """List research initiatives.

    Filter by status: proposed, planning, development, analysis, validation, documentation, published, cancelled.
    Filter by division: cancer, crohns, etc.
    Filter by labels (partial match): drug-candidate, novel-finding, biomarker, genomic, clinical, high-priority, etc.
    """
    qp: dict[str, str | int] = {"limit": limit}
    if status:
        qp["status"] = status
    if division:
        qp["division"] = division
    if labels:
        qp["labels"] = labels
    return json.dumps(_api("GET", f"/projects?{urlencode(qp)}"), indent=2)


@mcp.tool()
def get_project(name: str) -> str:
    """Get a specific research initiative by name."""
    return json.dumps(_api("GET", f"/projects/{quote(name, safe='')}"), indent=2)


@mcp.tool()
def create_project(
    name: str,
    description: str = "",
    division: str = "",
    status: str = "proposed",
    labels: str = "",
    plan_content: str = "",
    findings_content: str = "",
    plain_summary: str = "",
    impact_score: str = "",
) -> str:
    """Create a new research initiative in the coordination API.

    Division: research division (e.g. cancer, crohns).
    Labels are comma-separated tags for categorizing research. Suggested labels:
    drug-candidate, novel-finding, biomarker, genomic, transcriptomic, clinical,
    drug-screening, resistance, protein, high-priority, promising

    plan_content: research plan text (shown in project detail view).
    findings_content: findings/summary text (shown in project detail view).
    plain_summary: a plain English summary of the research results written for
        non-scientists. Explain what was studied, what was found, and why it
        matters in everyday language. Avoid jargon. 2-4 sentences.
    impact_score: significance rating — one of: breakthrough, high, moderate,
        incremental. Use 'breakthrough' for novel findings that could change
        treatment approaches, 'high' for promising results with strong
        therapeutic potential, 'moderate' for useful contributions to the field,
        'incremental' for confirmatory or small-step results.
    """
    payload: dict = {"name": name, "status": status}
    if division:
        payload["division"] = division
    if description:
        payload["description"] = description
    if labels:
        payload["labels"] = labels
    if plan_content:
        payload["plan_content"] = plan_content
    if findings_content:
        payload["findings_content"] = findings_content
    if plain_summary:
        payload["plain_summary"] = plain_summary
    if impact_score:
        payload["impact_score"] = impact_score
    return json.dumps(_api("POST", "/projects", json=payload), indent=2)


@mcp.tool()
def update_project(
    name: str,
    status: str = "",
    description: str = "",
    labels: str = "",
    plan_content: str = "",
    findings_content: str = "",
    plain_summary: str = "",
    impact_score: str = "",
) -> str:
    """Update a research initiative's status, description, and/or labels.

    Labels are comma-separated tags: drug-candidate, novel-finding, biomarker,
    genomic, transcriptomic, clinical, drug-screening, resistance, protein,
    high-priority, promising

    plan_content: research plan text (shown in project detail view).
    findings_content: findings/summary text (shown in project detail view).
    plain_summary: a plain English summary of the research results written for
        non-scientists. Explain what was studied, what was found, and why it
        matters in everyday language. Avoid jargon. 2-4 sentences.
    impact_score: significance rating — one of: breakthrough, high, moderate,
        incremental. Use 'breakthrough' for novel findings that could change
        treatment approaches, 'high' for promising results with strong
        therapeutic potential, 'moderate' for useful contributions to the field,
        'incremental' for confirmatory or small-step results.
    """
    payload: dict = {}
    if status:
        payload["status"] = status
    if description:
        payload["description"] = description
    if labels:
        payload["labels"] = labels
    if plan_content:
        payload["plan_content"] = plan_content
    if findings_content:
        payload["findings_content"] = findings_content
    if plain_summary:
        payload["plain_summary"] = plain_summary
    if impact_score:
        payload["impact_score"] = impact_score
    if not payload:
        return json.dumps({"error": "no fields to update"})
    return json.dumps(
        _api("PATCH", f"/projects/{quote(name, safe='')}", json=payload), indent=2
    )


# ── Status & Presence Tools ──


@mcp.tool()
def get_status() -> str:
    """Get aggregate system status: running agents, active projects, task summary counts."""
    return json.dumps(_api("GET", "/status"), indent=2)


@mcp.tool()
def list_agents(status: str = "", division: str = "") -> str:
    """List agent presence. Filter by status: running, idle. Filter by division."""
    qp: dict[str, str] = {}
    if status:
        qp["status"] = status
    if division:
        qp["division"] = division
    path = f"/agents?{urlencode(qp)}" if qp else "/agents"
    return json.dumps(_api("GET", path), indent=2)


# ── GitHub Tools (operate on this repo) ──


@mcp.tool()
def list_issues(limit: int = 50) -> str:
    """List open GitHub issues for this repository."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    check = _gh("repo", "view", repo, "--json", "name")
    if "exit code" in check:
        return f"No repo: {repo}"

    output = _gh(
        "issue",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        str(limit),
        "--json",
        "number,title,labels,updatedAt,author,comments",
        "--template",
        '{{range .}}#{{.number}} [{{timeago .updatedAt}}] "{{.title}}" by:{{.author.login}} comments:{{len .comments}}{{range .labels}} [{{.name}}]{{end}}\n{{end}}',
    )
    return output if output else f"No open issues: {repo}"


@mcp.tool()
def list_prs(limit: int = 50) -> str:
    """List open GitHub pull requests for this repository."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    check = _gh("repo", "view", repo, "--json", "name")
    if "exit code" in check:
        return f"No repo: {repo}"

    output = _gh(
        "pr",
        "list",
        "--repo",
        repo,
        "--state",
        "open",
        "--limit",
        str(limit),
        "--json",
        "number,title,updatedAt,author,reviewDecision,mergeable,isDraft,additions,deletions",
        "--template",
        '{{range .}}#{{.number}} [{{timeago .updatedAt}}] "{{.title}}" by:{{.author.login}} +{{.additions}}/-{{.deletions}} mergeable:{{.mergeable}} review:{{.reviewDecision}}{{if .isDraft}} DRAFT{{end}}\n{{end}}',
    )
    return output if output else f"No open PRs: {repo}"


@mcp.tool()
def comment_issue(number: int, body: str) -> str:
    """Add a comment to a GitHub issue."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    return _gh("issue", "comment", str(number), "--repo", repo, "--body", body)


@mcp.tool()
def close_issue(number: int, comment: str = "") -> str:
    """Close a GitHub issue, optionally with a closing comment."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    args = ["issue", "close", str(number), "--repo", repo]
    if comment:
        args += ["--comment", comment]
    return _gh(*args)


_MERGE_STRATEGIES = {"squash", "merge", "rebase"}
_REVIEW_ACTIONS = {"approve", "request-changes", "comment"}


@mcp.tool()
def merge_pr(number: int, strategy: str = "squash") -> str:
    """Merge a GitHub pull request.

    Strategy: squash (default), merge, or rebase.
    """
    if strategy not in _MERGE_STRATEGIES:
        return json.dumps({"error": f"invalid strategy: {strategy!r}, must be one of {_MERGE_STRATEGIES}"})
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    return _gh("pr", "merge", str(number), f"--{strategy}", "--repo", repo)


@mcp.tool()
def review_pr(number: int, action: str = "comment", body: str = "") -> str:
    """Review a GitHub pull request.

    Action: approve, request-changes, or comment.
    """
    if action not in _REVIEW_ACTIONS:
        return json.dumps({"error": f"invalid action: {action!r}, must be one of {_REVIEW_ACTIONS}"})
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    args = ["pr", "review", str(number), "--repo", repo, f"--{action}"]
    if body:
        args += ["--body", body]
    return _gh(*args)


@mcp.tool()
def close_pr(number: int, comment: str = "") -> str:
    """Close a GitHub pull request, optionally with a comment."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    args = ["pr", "close", str(number), "--repo", repo]
    if comment:
        args += ["--comment", comment]
    return _gh(*args)


@mcp.tool()
def get_pr_diff(number: int) -> str:
    """Get the diff of a GitHub pull request."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    return _gh("pr", "diff", str(number), "--repo", repo)


@mcp.tool()
def get_pr_checks(number: int) -> str:
    """Get CI check status for a GitHub pull request."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    return _gh("pr", "checks", str(number), "--repo", repo)


@mcp.tool()
def create_release(version: str, generate_notes: bool = True, body: str = "") -> str:
    """Create a GitHub release.

    Version should include the 'v' prefix (e.g. 'v0.1.0').
    """
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    args = ["release", "create", version, "--repo", repo]
    if generate_notes:
        args.append("--generate-notes")
    if body:
        args += ["--notes", body]
    return _gh(*args)


@mcp.tool()
def list_ci_runs(limit: int = 10) -> str:
    """List recent CI workflow runs."""
    repo = f"{GITHUB_ORG}/{GITHUB_REPO}"
    return _gh("run", "list", "--repo", repo, "--limit", str(limit))


# ── Data Management Tools ──


@mcp.tool()
def list_data(subdir: str = "") -> str:
    """List files in the data directory. Optionally specify a subdirectory."""
    target = (DATA_DIR / subdir).resolve() if subdir else DATA_DIR.resolve()
    if not target.is_relative_to(DATA_DIR.resolve()):
        return "Error: path must be within data directory"
    if not target.is_dir():
        return f"Directory not found: {target}"
    entries = []
    for item in sorted(target.iterdir()):
        kind = "dir" if item.is_dir() else "file"
        size = item.stat().st_size if item.is_file() else ""
        entries.append({"name": item.name, "type": kind, "size": size})
    return json.dumps(entries, indent=2)


@mcp.tool()
def ensure_data_dir(subdir: str) -> str:
    """Create a subdirectory under data/ for organizing research data.

    Example: ensure_data_dir("tcga/brca") creates data/tcga/brca/
    """
    target = (DATA_DIR / subdir).resolve()
    if not target.is_relative_to(DATA_DIR.resolve()):
        return "Error: path must be within data directory"
    target.mkdir(parents=True, exist_ok=True)
    return f"Directory ready: {target}"


if __name__ == "__main__":
    mcp.run()
