import io
import zipfile
from datetime import datetime, timezone
from html import escape as _esc
from urllib.parse import quote, urlencode

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import func, select, union
from sqlalchemy.orm import Session

from bioagentics.agent_api.database import (
    SessionLocal,
    agents,
    api_keys,
    journal,
    projects_table,
    tasks,
)
from bioagentics.config import REPO_ROOT

router = APIRouter(tags=["ui"])

PER_PAGE = 30


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Utilities ─────────────────────────────────────────────────────────


def esc(value) -> str:
    if value is None:
        return ""
    return _esc(str(value))


def time_ago(iso_str) -> str:
    if not iso_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        diff = (datetime.now(timezone.utc) - dt).total_seconds()
        if diff < 60:
            return "just now"
        if diff < 3600:
            return f"{int(diff // 60)}m ago"
        if diff < 86400:
            return f"{int(diff // 3600)}h ago"
        return f"{int(diff // 86400)}d ago"
    except Exception:
        return str(iso_str)


def time_tag(iso_str) -> str:
    """Render a timestamp as a styled tooltip span."""
    ago = time_ago(iso_str)
    if not ago:
        return ""
    return f'<span class="timestamp text-xs text-[#a1a1aa] cursor-default" data-ts="{esc(iso_str)}">{ago}</span>'


def is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


def build_qs(**params) -> str:
    filtered = {k: v for k, v in params.items() if v is not None and v != ""}
    return urlencode(filtered, quote_via=quote) if filtered else ""


def check_api_key(db: Session, key: str) -> bool:
    has_keys = db.execute(select(api_keys.c.id).limit(1)).first()
    if not has_keys:
        return True  # no keys = open access
    if not key:
        return False
    row = db.execute(select(api_keys).where(api_keys.c.key == key)).first()
    return row is not None


# ── SVG Icons ─────────────────────────────────────────────────────────

ICONS = {
    "chevron-down": '<svg class="w-4 h-4 pointer-events-none" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg>',
    "x": '<svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M18 6L6 18M6 6l12 12"/></svg>',
    "inbox": '<svg class="w-10 h-10 text-[#3f3f46]" fill="none" stroke="currentColor" stroke-width="1.5" viewBox="0 0 24 24"><path d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7"/><path d="M2 13h6l2 3h4l2-3h6v5a2 2 0 01-2 2H4a2 2 0 01-2-2v-5z"/></svg>',
    "dashboard": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>',
    "folder": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"/></svg>',
    "tasks": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>',
    "journal": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M2 3h6a4 4 0 014 4v14a3 3 0 00-3-3H2z"/><path d="M22 3h-6a4 4 0 00-4 4v14a3 3 0 013-3h7z"/></svg>',
    "key": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 2l-2 2m-7.61 7.61a5.5 5.5 0 11-7.778 7.778 5.5 5.5 0 017.777-7.777zm0 0L15.5 7.5m0 0l3 3L22 7l-3-3m-3.5 3.5L19 4"/></svg>',
    "play": '<svg class="w-4 h-4 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "search": '<svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>',
    "external": '<svg class="w-3 h-3 inline-block" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M18 13v6a2 2 0 01-2 2H5a2 2 0 01-2-2V8a2 2 0 012-2h6"/><polyline points="15 3 21 3 21 9"/><line x1="10" y1="14" x2="21" y2="3"/></svg>',
    "alert": '<svg class="w-5 h-5 shrink-0" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "menu": '<svg class="w-5 h-5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>',
    "github": '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>',
    "chart": '<svg class="w-7 h-7" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>',
    "book": '<svg class="w-7 h-7" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 19.5A2.5 2.5 0 016.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 014 19.5v-15A2.5 2.5 0 016.5 2z"/></svg>',
}


# ── Status / Priority helpers ─────────────────────────────────────────


STATUS_COLORS = {
    "pending": ("bg-[#27272a]", "text-[#a1a1aa]"),
    "in_progress": ("bg-[#052e16]", "text-[#4ade80]"),
    "blocked": ("bg-[#422006]", "text-[#fbbf24]"),
    "done": ("bg-[#172554]", "text-[#60a5fa]"),
    "cancelled": ("bg-[#450a0a]", "text-[#f87171]"),
}

PROJECT_STATUS_COLORS = {
    "proposed": ("bg-[#27272a]", "text-[#a1a1aa]"),
    "planning": ("bg-[#2e1065]", "text-[#a78bfa]"),
    "development": ("bg-[#172554]", "text-[#60a5fa]"),
    "analysis": ("bg-[#422006]", "text-[#fbbf24]"),
    "validation": ("bg-[#3b0764]", "text-[#c084fc]"),
    "documentation": ("bg-[#164e63]", "text-[#67e8f9]"),
    "published": ("bg-[#052e16] border border-[#166534]", "text-[#4ade80]"),
    "cancelled": ("bg-[#450a0a]", "text-[#f87171]"),
}

AGENT_DISPLAY_NAMES = {
    "research_director": "Research Director",
    "literature_reviewer": "Literature Reviewer",
    "data_curator": "Data Curator",
    "project_manager": "Project Manager",
    "developer": "Developer",
    "analyst": "Analyst",
    "validation_scientist": "Validation Scientist",
    "research_writer": "Research Writer",
    "human": "Human",
    "dispatcher": "Dispatcher",
}


def agent_display_name(username: str) -> str:
    return AGENT_DISPLAY_NAMES.get(username, username.replace("_", " ").title())


PRIORITY_COLORS = {
    1: "text-[#71717a]",
    2: "text-[#a1a1aa]",
    3: "text-[#fbbf24]",
    4: "text-[#f97316]",
    5: "text-[#ef4444]",
}


def status_badge(status: str) -> str:
    bg, fg = STATUS_COLORS.get(status, ("bg-[#27272a]", "text-[#a1a1aa]"))
    return f'<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium {bg} {fg}">{esc(status)}</span>'


def project_status_badge(status: str) -> str:
    bg, fg = PROJECT_STATUS_COLORS.get(status, ("bg-[#27272a]", "text-[#a1a1aa]"))
    return f'<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium {bg} {fg}">{esc(status)}</span>'


def labels_tags(labels: str | None) -> str:
    if not labels:
        return '<span class="text-[#a1a1aa] text-xs">-</span>'
    parts = []
    for label in labels.split(","):
        label = label.strip()
        if label:
            parts.append(f'<span class="inline-flex items-center px-1.5 py-0.5 rounded-full text-[11px] font-medium bg-[#1e1b4b] text-[#818cf8]">{esc(label)}</span>')
    return f'<span class="inline-flex gap-1 flex-wrap">{"".join(parts)}</span>' if parts else '<span class="text-[#a1a1aa] text-xs">-</span>'


def priority_label(p: int) -> str:
    color = PRIORITY_COLORS.get(p, "text-[#a1a1aa]")
    return f'<span class="font-semibold text-xs tabular-nums {color}">P{p}</span>'


# ── Component Renderers ──────────────────────────────────────────────


def render_task_card(row) -> str:
    m = row._mapping
    is_human = m["username"] == "human"
    can_act = is_human and m["status"] not in ("done", "cancelled")

    action_btns = ""
    if can_act:
        action_btns = f"""
  <div class="flex gap-2 mt-3 pt-3 border-t border-[#27272a]">
    <button class="bg-[#052e16] text-[#4ade80] border border-[#166534] rounded-md px-2.5 py-1 text-xs font-medium cursor-pointer whitespace-nowrap hover:bg-[#14532d] h-8 transition-colors"
            onclick="openActionModal({m['id']},'done')">Mark Done</button>
    <button class="bg-[#450a0a] text-[#f87171] border border-[#7f1d1d] rounded-md px-2.5 py-1 text-xs font-medium cursor-pointer whitespace-nowrap hover:bg-[#7f1d1d] h-8 transition-colors"
            onclick="openActionModal({m['id']},'cancelled')">Reject</button>
  </div>"""

    project_html = ""
    if m["project"]:
        pq = build_qs(project=m["project"])
        project_html = f"""<a class="text-[#3b82f6] text-xs no-underline hover:underline cursor-pointer"
               hx-get="/ui/tasks?{pq}" hx-target="#tab-content" hx-push-url="true">{esc(m['project'])}</a>"""

    desc_html = ""
    if m.get("description"):
        desc_html = f'<div class="text-sm whitespace-pre-wrap break-words mt-1.5 text-[#a1a1aa] line-clamp-3">{esc(m["description"])}</div>'

    return f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4
                    hover:border-[#3f3f46] transition-colors flex flex-col" id="task-{m['id']}">
  <div class="flex items-center gap-1.5 mb-1.5 flex-wrap">
    <span class="text-[#71717a] text-xs font-mono cursor-pointer hover:text-[#3b82f6] transition-colors"
          onclick="navigator.clipboard.writeText('#{m['id']}')">#{m['id']}</span>
    {priority_label(m['priority'])}
    {status_badge(m['status'])}
  </div>
  <a class="text-sm font-medium leading-snug line-clamp-2 no-underline text-[#fafafa] hover:text-[#3b82f6] transition-colors cursor-pointer"
     hx-get="/ui/tasks/{m['id']}" hx-target="#tab-content" hx-push-url="true">{esc(m['title'])}</a>
  {desc_html}
  <div class="mt-auto pt-2.5 flex items-center justify-between text-xs text-[#71717a]">
    <div class="flex items-center gap-2 min-w-0">
      <span class="text-[#a78bfa] font-medium shrink-0">{esc(agent_display_name(m['username']))}</span>
      {project_html}
    </div>
    {time_tag(m['updated_at'])}
  </div>
  {action_btns}
</div>"""


def render_task_compact(row) -> str:
    """Abbreviated task row for dashboard widget."""
    m = row._mapping
    project_html = ""
    if m["project"]:
        project_html = f'<span class="text-[#a1a1aa]">{esc(m["project"])}</span>'
    return f"""<div class="flex items-center gap-3 py-2 border-b border-[#27272a] last:border-b-0">
  <span class="text-[#a1a1aa] text-xs font-mono shrink-0">#{m['id']}</span>
  {priority_label(m['priority'])}
  {status_badge(m['status'])}
  <span class="text-sm truncate flex-1 min-w-0">{esc(m['title'])}</span>
  <span class="text-[#a78bfa] text-xs font-semibold shrink-0">{esc(agent_display_name(m['username']))}</span>
  {project_html}
</div>"""


def render_journal_card(row) -> str:
    m = row._mapping
    project_html = ""
    if m["project"]:
        pq = build_qs(project=m["project"])
        project_html = f"""<a class="text-[#3b82f6] text-xs no-underline hover:underline cursor-pointer"
               hx-get="/ui/journal?{pq}" hx-target="#tab-content" hx-push-url="true">{esc(m['project'])}</a>"""

    return f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4
                    flex flex-col hover:border-[#3f3f46] transition-colors cursor-pointer"
                    hx-get="/ui/journal/{m['id']}" hx-target="#tab-content" hx-push-url="true">
  <div class="flex items-center gap-2 mb-1.5 text-xs text-[#71717a] flex-wrap">
    <span class="font-mono">#{m['id']}</span>
    <span class="text-[#a78bfa] font-medium">{esc(agent_display_name(m['username']))}</span>
    {project_html}
    <span class="ml-auto shrink-0">{time_tag(m['created_at'])}</span>
  </div>
  <div class="text-sm whitespace-pre-wrap break-words line-clamp-6 flex-1 text-[#d4d4d8]">{esc(m['content'])}</div>
</div>"""


def render_journal_compact(row) -> str:
    """Abbreviated journal row for dashboard widget."""
    m = row._mapping
    project_html = ""
    if m["project"]:
        project_html = f'<span class="text-[#a1a1aa]">{esc(m["project"])}</span>'
    # Truncate content to first line
    content = str(m["content"] or "")
    first_line = content.split("\n")[0][:120]
    if len(first_line) < len(content):
        first_line += "..."
    return f"""<div class="flex items-center gap-3 py-2 border-b border-[#27272a] last:border-b-0">
  <span class="text-[#a1a1aa] text-xs font-mono shrink-0">#{m['id']}</span>
  <span class="text-[#a78bfa] text-xs font-semibold shrink-0">{esc(agent_display_name(m['username']))}</span>
  {project_html}
  <span class="text-sm text-[#a1a1aa] truncate flex-1 min-w-0">{esc(first_line)}</span>
  <span class="text-xs text-[#a1a1aa] shrink-0" title="{esc(m['created_at'])}">{time_ago(m['created_at'])}</span>
</div>"""


def render_task_detail(row) -> str:
    """Full detail view for a single task."""
    m = row._mapping
    is_human = m["username"] == "human"
    can_act = is_human and m["status"] not in ("done", "cancelled")

    action_btns = ""
    if can_act:
        action_btns = f"""
  <div class="flex gap-2 mt-4">
    <button class="bg-[#052e16] text-[#4ade80] border border-[#166534] rounded-md px-3 py-1.5 text-xs font-medium cursor-pointer hover:bg-[#14532d] transition-colors"
            onclick="openActionModal({m['id']},'done')">Mark Done</button>
    <button class="bg-[#450a0a] text-[#f87171] border border-[#7f1d1d] rounded-md px-3 py-1.5 text-xs font-medium cursor-pointer hover:bg-[#7f1d1d] transition-colors"
            onclick="openActionModal({m['id']},'cancelled')">Reject</button>
  </div>"""

    project_html = ""
    if m["project"]:
        pq = build_qs(project=m["project"])
        project_html = f"""<a class="text-[#3b82f6] text-sm no-underline hover:underline cursor-pointer"
               hx-get="/ui/tasks?{pq}" hx-target="#tab-content" hx-push-url="true">{esc(m['project'])}</a>"""

    desc_html = ""
    if m.get("description"):
        desc_html = f"""<div class="mt-4">
    <div class="text-xs text-[#71717a] font-medium mb-1.5">Description</div>
    <div class="text-sm whitespace-pre-wrap break-words text-[#d4d4d8] bg-[#09090b] rounded-lg p-4 border border-[#27272a]">{esc(m['description'])}</div>
  </div>"""

    blocked_html = ""
    if m.get("blocked_reason"):
        blocked_html = f"""<div class="mt-4">
    <div class="text-xs text-[#71717a] font-medium mb-1.5">Blocked Reason</div>
    <div class="text-sm whitespace-pre-wrap break-words text-[#fbbf24] bg-[#422006]/30 rounded-lg p-4 border border-[#854d0e]/40">{esc(m['blocked_reason'])}</div>
  </div>"""

    return f"""<div class="mb-4">
  <a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors"
     hx-get="/ui/tasks" hx-target="#tab-content" hx-push-url="true">&larr; Back to tasks</a>
</div>
<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 max-w-3xl">
  <div class="flex items-center gap-2 mb-3 flex-wrap">
    <span class="text-[#71717a] text-xs font-mono">#{m['id']}</span>
    {priority_label(m['priority'])}
    {status_badge(m['status'])}
  </div>
  <h2 class="text-lg font-semibold mb-3">{esc(m['title'])}</h2>
  <div class="flex items-center gap-3 text-sm text-[#a1a1aa]">
    <span class="text-[#a78bfa] font-medium">{esc(agent_display_name(m['username']))}</span>
    {project_html}
  </div>
  {desc_html}
  {blocked_html}
  <div class="mt-4 pt-4 border-t border-[#27272a] flex items-center gap-4 text-xs text-[#71717a]">
    <span>Created {time_tag(m['created_at'])}</span>
    <span>Updated {time_tag(m['updated_at'])}</span>
  </div>
  {action_btns}
</div>"""


def render_journal_detail(row) -> str:
    """Full detail view for a single journal entry."""
    m = row._mapping

    project_html = ""
    if m["project"]:
        pq = build_qs(project=m["project"])
        project_html = f"""<a class="text-[#3b82f6] text-sm no-underline hover:underline cursor-pointer"
               hx-get="/ui/journal?{pq}" hx-target="#tab-content" hx-push-url="true">{esc(m['project'])}</a>"""

    return f"""<div class="mb-4">
  <a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors"
     hx-get="/ui/journal" hx-target="#tab-content" hx-push-url="true">&larr; Back to journal</a>
</div>
<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 max-w-3xl">
  <div class="flex items-center gap-2 mb-3 text-sm text-[#71717a] flex-wrap">
    <span class="font-mono text-xs">#{m['id']}</span>
    <span class="text-[#a78bfa] font-medium">{esc(agent_display_name(m['username']))}</span>
    {project_html}
    <span class="ml-auto">{time_tag(m['created_at'])}</span>
  </div>
  <div class="text-sm whitespace-pre-wrap break-words text-[#d4d4d8]">{esc(m['content'])}</div>
</div>"""


def render_project_detail(
    row, task_counts: dict, journal_count: int, recent_tasks, recent_journal,
    *, plan_content: str | None = None, findings_content: str | None = None,
    result_files: list[str] | None = None, output_files: list[str] | None = None,
) -> str:
    """Full detail view for a single project."""
    m = row._mapping
    name = m["name"]
    tc = task_counts
    pend = tc.get("pending", 0)
    act = tc.get("in_progress", 0)
    blk = tc.get("blocked", 0)
    done = tc.get("done", 0)
    cancelled = tc.get("cancelled", 0)
    total = pend + act + blk + done + cancelled

    desc_html = ""
    if m.get("description"):
        desc_html = f"""<div class="mt-4">
    <div class="text-xs text-[#71717a] font-medium mb-1.5">Description</div>
    <div class="text-sm whitespace-pre-wrap break-words text-[#d4d4d8] bg-[#09090b] rounded-lg p-4 border border-[#27272a]">{esc(m['description'])}</div>
  </div>"""

    # Task summary bar
    task_summary = '<span class="text-[#a1a1aa] text-sm">No tasks yet</span>'
    if total > 0:
        parts = []
        for count, label, bg, fg in [
            (pend, "pending", "bg-[#27272a]", "text-[#a1a1aa]"),
            (act, "in progress", "bg-[#052e16]", "text-[#4ade80]"),
            (blk, "blocked", "bg-[#3d2a1a]", "text-[#fbbf24]"),
            (done, "done", "bg-[#172554]", "text-[#3b82f6]"),
            (cancelled, "cancelled", "bg-[#27272a]", "text-[#71717a]"),
        ]:
            if count:
                parts.append(f'<span class="{bg} {fg} px-2 py-0.5 rounded-full text-xs font-medium">{count} {label}</span>')
        task_summary = f'<div class="flex gap-2 flex-wrap">{"".join(parts)}</div>'

    # Recent tasks list
    tasks_html = ""
    if recent_tasks:
        task_rows = ""
        for t in recent_tasks:
            tm = t._mapping
            task_rows += f"""<a class="flex items-center gap-2 py-2 border-b border-[#27272a] last:border-b-0 no-underline hover:bg-[#18181b] px-2 -mx-2 rounded cursor-pointer transition-colors"
   hx-get="/ui/tasks/{tm['id']}" hx-target="#tab-content" hx-push-url="true">
  <span class="text-[#71717a] text-xs font-mono shrink-0">#{tm['id']}</span>
  {priority_label(tm['priority'])}
  {status_badge(tm['status'])}
  <span class="text-sm truncate flex-1 min-w-0">{esc(tm['title'])}</span>
  <span class="text-[#a78bfa] text-xs shrink-0">{esc(agent_display_name(tm['username']))}</span>
</a>"""
        tq = build_qs(project=name)
        tasks_html = f"""<div class="mt-6">
  <div class="flex items-center justify-between mb-2">
    <div class="text-xs text-[#71717a] font-medium">Recent Tasks</div>
    <a class="text-xs text-[#3b82f6] no-underline hover:underline cursor-pointer"
       hx-get="/ui/tasks?{tq}" hx-target="#tab-content" hx-push-url="true">View all &rarr;</a>
  </div>
  <div>{task_rows}</div>
</div>"""

    # Recent journal entries
    journal_html = ""
    if recent_journal:
        journal_rows = ""
        for j in recent_journal:
            jm = j._mapping
            content = str(jm["content"] or "")
            first_line = content.split("\n")[0][:120]
            if len(first_line) < len(content):
                first_line += "..."
            journal_rows += f"""<a class="flex items-center gap-2 py-2 border-b border-[#27272a] last:border-b-0 no-underline hover:bg-[#18181b] px-2 -mx-2 rounded cursor-pointer transition-colors"
   hx-get="/ui/journal/{jm['id']}" hx-target="#tab-content" hx-push-url="true">
  <span class="text-[#71717a] text-xs font-mono shrink-0">#{jm['id']}</span>
  <span class="text-[#a78bfa] text-xs font-semibold shrink-0">{esc(agent_display_name(jm['username']))}</span>
  <span class="text-sm text-[#a1a1aa] truncate flex-1 min-w-0">{esc(first_line)}</span>
  <span class="text-xs text-[#a1a1aa] shrink-0">{time_ago(jm['created_at'])}</span>
</a>"""
        jq = build_qs(project=name)
        journal_html = f"""<div class="mt-6">
  <div class="flex items-center justify-between mb-2">
    <div class="text-xs text-[#71717a] font-medium">Recent Journal Entries</div>
    <a class="text-xs text-[#3b82f6] no-underline hover:underline cursor-pointer"
       hx-get="/ui/journal?{jq}" hx-target="#tab-content" hx-push-url="true">View all &rarr;</a>
  </div>
  <div>{journal_rows}</div>
</div>"""

    # Research artifacts
    artifacts_html = ""
    artifact_sections = []

    if plan_content:
        artifact_sections.append(f"""<details class="group">
  <summary class="text-xs text-[#71717a] font-medium cursor-pointer hover:text-[#3b82f6] transition-colors select-none list-none flex items-center gap-1">
    <svg class="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>
    Research Plan
  </summary>
  <div class="text-sm whitespace-pre-wrap break-words text-[#d4d4d8] bg-[#09090b] rounded-lg p-4 border border-[#27272a] mt-2 max-h-[500px] overflow-y-auto">{esc(plan_content)}</div>
</details>""")

    if findings_content:
        artifact_sections.append(f"""<details open class="group">
  <summary class="text-xs text-[#71717a] font-medium cursor-pointer hover:text-[#3b82f6] transition-colors select-none list-none flex items-center gap-1">
    <svg class="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>
    Findings &amp; Summary
  </summary>
  <div class="text-sm whitespace-pre-wrap break-words text-[#d4d4d8] bg-[#09090b] rounded-lg p-4 border border-[#27272a] mt-2 max-h-[500px] overflow-y-auto">{esc(findings_content)}</div>
</details>""")

    if result_files:
        file_items = "".join(
            f'<div class="text-sm text-[#a1a1aa] py-1 font-mono text-xs">{esc(f)}</div>'
            for f in result_files
        )
        artifact_sections.append(f"""<details class="group">
  <summary class="text-xs text-[#71717a] font-medium cursor-pointer hover:text-[#3b82f6] transition-colors select-none list-none flex items-center gap-1">
    <svg class="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>
    Result Files ({len(result_files)})
  </summary>
  <div class="bg-[#09090b] rounded-lg p-4 border border-[#27272a] mt-2 max-h-[300px] overflow-y-auto">{file_items}</div>
</details>""")

    if output_files:
        ofile_items = "".join(
            f'<div class="text-sm text-[#a1a1aa] py-1 font-mono text-xs">{esc(f)}</div>'
            for f in output_files
        )
        download_url = f"/ui/projects/{quote(name, safe='')}/download"
        artifact_sections.append(f"""<details open class="group">
  <summary class="text-xs text-[#71717a] font-medium cursor-pointer hover:text-[#3b82f6] transition-colors select-none list-none flex items-center gap-1">
    <svg class="w-3 h-3 transition-transform group-open:rotate-90" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 18l6-6-6-6"/></svg>
    Output Files ({len(output_files)})
  </summary>
  <div class="bg-[#09090b] rounded-lg p-4 border border-[#27272a] mt-2">
    <div class="flex items-center justify-between mb-3">
      <span class="text-xs text-[#71717a]">{len(output_files)} file{'s' if len(output_files) != 1 else ''}</span>
      <a href="{download_url}" class="inline-flex items-center gap-1.5 px-3 py-1.5 bg-[#172554] text-[#3b82f6] text-xs font-medium rounded-md hover:bg-[#1e3a8a] transition-colors no-underline cursor-pointer">
        <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
        Download ZIP
      </a>
    </div>
    <div class="max-h-[250px] overflow-y-auto">{ofile_items}</div>
  </div>
</details>""")

    if artifact_sections:
        artifacts_html = f"""<div class="mt-6">
  <div class="text-xs text-[#71717a] font-medium mb-2">Research Artifacts</div>
  <div class="flex flex-col gap-3">{"".join(artifact_sections)}</div>
</div>"""

    return f"""<div class="mb-4">
  <a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors"
     hx-get="/ui/projects" hx-target="#tab-content" hx-push-url="true">&larr; Back to projects</a>
</div>
<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 max-w-3xl">
  <div class="flex items-center gap-2 mb-3 flex-wrap">
    {project_status_badge(m['status'])}
    {labels_tags(m.get('labels', ''))}
  </div>
  <h2 class="text-lg font-semibold mb-3">{esc(name)}</h2>
  {desc_html}
  {artifacts_html}
  <div class="mt-4">
    <div class="text-xs text-[#71717a] font-medium mb-1.5">Tasks ({total})</div>
    {task_summary}
  </div>
  {tasks_html}
  {journal_html}
  <div class="mt-6 pt-4 border-t border-[#27272a] flex items-center gap-4 text-xs text-[#71717a]">
    <span>Created {time_tag(m['created_at'])}</span>
    <span>Updated {time_tag(m['updated_at'])}</span>
  </div>
</div>"""


def render_project_row(p, task_counts: dict, journal_count: int) -> str:
    m = p._mapping
    tc = task_counts
    pend = tc.get("pending", 0)
    act = tc.get("in_progress", 0)
    blk = tc.get("blocked", 0)
    done = tc.get("done", 0)
    total = pend + act + blk + done + tc.get("cancelled", 0)

    name = m["name"]
    tc_html = '<span class="text-[#a1a1aa] text-xs">-</span>'
    if total > 0:
        parts = []
        if pend:
            pq = build_qs(project=name, status="pending")
            parts.append(
                f'<span class="bg-[#27272a] text-[#a1a1aa] px-1 py-0.5 rounded cursor-pointer hover:opacity-80"'
                f' hx-get="/ui/tasks?{pq}" hx-target="#tab-content" hx-push-url="true">{pend}p</span>'
            )
        if act:
            aq = build_qs(project=name, status="in_progress")
            parts.append(
                f'<span class="bg-[#052e16] text-[#4ade80] px-1 py-0.5 rounded cursor-pointer hover:opacity-80"'
                f' hx-get="/ui/tasks?{aq}" hx-target="#tab-content" hx-push-url="true">{act}a</span>'
            )
        if blk:
            bq = build_qs(project=name, status="blocked")
            parts.append(
                f'<span class="bg-[#3d2a1a] text-[#fbbf24] px-1 py-0.5 rounded cursor-pointer hover:opacity-80"'
                f' hx-get="/ui/tasks?{bq}" hx-target="#tab-content" hx-push-url="true">{blk}b</span>'
            )
        if done:
            dq = build_qs(project=name, status="done")
            parts.append(
                f'<span class="bg-[#172554] text-[#3b82f6] px-1 py-0.5 rounded cursor-pointer hover:opacity-80"'
                f' hx-get="/ui/tasks?{dq}" hx-target="#tab-content" hx-push-url="true">{done}d</span>'
            )
        tc_html = f'<span class="text-[11px] inline-flex gap-1 flex-wrap">{"".join(parts)}</span>'

    jc_html = '<span class="text-[#a1a1aa] text-xs">-</span>'
    if journal_count > 0:
        jq = build_qs(project=name)
        jc_html = (
            f'<span class="text-[#a1a1aa] text-xs cursor-pointer hover:text-[#3b82f6] hover:underline"'
            f' hx-get="/ui/journal?{jq}" hx-target="#tab-content" hx-push-url="true">{journal_count}</span>'
        )

    desc_html = ""
    if m.get("description"):
        desc_html = f'<div class="text-[#a1a1aa] text-xs max-w-[200px] overflow-hidden text-ellipsis whitespace-nowrap" title="{esc(m["description"])}">{esc(m["description"])}</div>'

    return f"""<tr class="hover:bg-[#18181b]">
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle">
    <a class="font-semibold text-[#3b82f6] text-sm no-underline hover:underline cursor-pointer"
       hx-get="/ui/projects/{quote(name)}" hx-target="#tab-content" hx-push-url="true">{esc(name)}</a>
    {desc_html}
  </td>
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle">{labels_tags(m.get('labels', ''))}</td>
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle">{project_status_badge(m['status'])}</td>
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle">{tc_html}</td>
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle">{jc_html}</td>
  <td class="px-2.5 py-2 border-b border-[#27272a] align-middle whitespace-nowrap">{time_tag(m['updated_at'])}</td>
</tr>"""


def render_key_row(row) -> str:
    m = row._mapping
    return f"""<div class="flex items-center gap-3 px-4 py-2.5 border-b border-[#27272a] last:border-b-0">
  <span class="font-semibold text-sm min-w-[120px]">{esc(m['name'])}</span>
  <span class="font-mono text-sm text-[#a1a1aa] bg-[#09090b] px-2 py-0.5 rounded">{esc(m['key'])}</span>
  <span class="ml-auto">{time_tag(m['created_at'])}</span>
  <button class="bg-[#450a0a] text-[#f87171] border border-[#7f1d1d] rounded-lg px-2.5 py-1.5 text-xs font-semibold cursor-pointer whitespace-nowrap hover:bg-[#7f1d1d] min-h-[44px] transition-colors"
          data-key-id="{m['id']}" data-key-name="{esc(m['name'])}"
          onclick="revokeKey(+this.dataset.keyId, this.dataset.keyName)">Revoke</button>
</div>"""


def render_pager(prefix: str, path: str, offset: int, total: int, **params) -> str:
    page = offset // PER_PAGE + 1
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)

    prev_disabled = "opacity-40 cursor-default pointer-events-none" if offset == 0 else ""
    next_disabled = (
        "opacity-40 cursor-default pointer-events-none"
        if offset + PER_PAGE >= total
        else ""
    )

    prev_offset = max(0, offset - PER_PAGE)
    next_offset = offset + PER_PAGE

    prev_qs = build_qs(offset=prev_offset, **params) if offset > 0 else ""
    next_qs = build_qs(offset=next_offset, **params)

    prev_label = "&larr; Newer" if prefix == "j" else "&larr; Prev"
    next_label = "Older &rarr;" if prefix == "j" else "Next &rarr;"

    return f"""<div class="flex gap-2 justify-center items-center mt-4">
  <button class="bg-[#09090b] border border-[#27272a] text-[#fafafa] px-4 py-2 rounded-md text-sm cursor-pointer min-h-[36px] hover:bg-[#0c0c0e] transition-colors {prev_disabled}"
          hx-get="{path}?{prev_qs}" hx-target="#tab-content" hx-push-url="true">{prev_label}</button>
  <span class="text-sm text-[#a1a1aa]">{page} / {pages}</span>
  <button class="bg-[#09090b] border border-[#27272a] text-[#fafafa] px-4 py-2 rounded-md text-sm cursor-pointer min-h-[36px] hover:bg-[#0c0c0e] transition-colors {next_disabled}"
          hx-get="{path}?{next_qs}" hx-target="#tab-content" hx-push-url="true">{next_label}</button>
</div>"""


# ── Filter Bar Renderers ─────────────────────────────────────────────

INPUT_CLS = (
    "bg-[#09090b] border border-[#27272a] text-[#fafafa] px-3 py-2 rounded-md text-sm"
    " placeholder:text-[#52525b] min-h-[36px]"
    " focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6]"
    " transition-colors"
)
SELECT_CLS = (
    "bg-[#09090b] border border-[#27272a] text-[#fafafa] pl-3 pr-8 py-2 rounded-md text-sm"
    " min-h-[36px] cursor-pointer appearance-none"
    " focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6]"
    " transition-colors"
)


def _search_input(name: str, value: str, placeholder: str, path: str) -> str:
    active_border = " !border-[#3b82f6]" if value else ""
    return f"""<div class="relative w-full sm:w-[220px]">
  <span class="absolute left-2.5 top-1/2 -translate-y-1/2 text-[#52525b] pointer-events-none">{ICONS["search"]}</span>
  <input name="{name}" value="{esc(value)}" placeholder="{placeholder}" autocomplete="off"
         class="{INPUT_CLS} w-full !pl-8{active_border}"
         hx-get="{path}" hx-target="#tab-content" hx-trigger="input changed delay:300ms"
         hx-include="closest form" hx-push-url="true" />
</div>"""


def _select(name: str, value: str, options: list[tuple[str, str]], path: str) -> str:
    opts = []
    for val, label in options:
        sel = " selected" if val == value else ""
        opts.append(f'<option value="{esc(val)}"{sel}>{esc(label)}</option>')
    active_border = " !border-[#3b82f6]" if value else ""
    return f"""<div class="relative inline-block">
  <select name="{name}" class="{SELECT_CLS}{active_border}"
       hx-get="{path}" hx-target="#tab-content" hx-trigger="change"
       hx-include="closest form" hx-push-url="true">
  {"".join(opts)}
  </select>
  <span class="absolute right-2 top-1/2 -translate-y-1/2 text-[#52525b] pointer-events-none">{ICONS["chevron-down"]}</span>
</div>"""


def _text_input(
    name: str, value: str, placeholder: str, path: str, datalist: str = ""
) -> str:
    dl = f' list="{datalist}"' if datalist else ""
    extra = ' data-1p-ignore data-lpignore="true" data-form-type="other"' if name in ("username",) else ""
    active_border = " !border-[#3b82f6]" if value else ""
    return f"""<input name="{name}" value="{esc(value)}" placeholder="{placeholder}" autocomplete="off"
       class="{INPUT_CLS}{active_border}"{dl}{extra}
       hx-get="{path}" hx-target="#tab-content" hx-trigger="input changed delay:300ms"
       hx-include="closest form" hx-push-url="true" />"""


def _clear_filters_btn(path: str, has_filters: bool) -> str:
    if not has_filters:
        return ""
    return f"""<a class="inline-flex items-center gap-1 text-xs text-[#a1a1aa] hover:text-[#fafafa] cursor-pointer transition-colors px-2 py-1.5 rounded-md hover:bg-[#0c0c0e] min-h-[36px] no-underline"
       hx-get="{path}" hx-target="#tab-content" hx-push-url="true">
  {ICONS["x"]} Clear filters
</a>"""


# ── Tab Content Renderers ─────────────────────────────────────────────


def render_dashboard_tab(db: Session, division: str = "") -> str:
    """Dashboard home view with summary widgets."""
    # Recent active tasks
    task_q = select(tasks).where(tasks.c.status.in_(["pending", "in_progress", "blocked"]))
    if division:
        task_q = task_q.where(tasks.c.division == division)
    recent_tasks = db.execute(
        task_q.order_by(tasks.c.priority.desc(), tasks.c.updated_at.desc()).limit(8)
    ).fetchall()

    # Recent journal
    journal_q = select(journal)
    if division:
        journal_q = journal_q.where(journal.c.division == division)
    recent_journal = db.execute(
        journal_q.order_by(journal.c.created_at.desc()).limit(5)
    ).fetchall()

    # Active agents
    active_agents = db.execute(
        select(agents).order_by(agents.c.updated_at.desc())
    ).fetchall()

    # Human action needed
    human_tasks = db.execute(
        select(tasks)
        .where(tasks.c.username == "human")
        .where(tasks.c.status.in_(["pending", "in_progress", "blocked"]))
        .order_by(tasks.c.priority.desc())
    ).fetchall()

    # Human alert banner
    human_banner = ""
    if human_tasks:
        count = len(human_tasks)
        hq = build_qs(username="human", status="pending")
        human_banner = f"""<div class="lg:col-span-2 bg-[#422006]/50 border border-[#a16207] rounded-lg px-4 py-3 flex items-center gap-3 cursor-pointer hover:bg-[#422006]/70 transition-colors"
     hx-get="/ui/tasks?{hq}" hx-target="#tab-content" hx-push-url="true">
  <span class="text-[#fbbf24]">{ICONS["alert"]}</span>
  <div class="flex-1">
    <span class="text-sm font-medium text-[#fbbf24]">{count} task{"s" if count != 1 else ""} need{"" if count != 1 else "s"} human action</span>
    <span class="text-xs text-[#a1a1aa] ml-2">Click to view</span>
  </div>
</div>"""

    # Tasks widget
    if recent_tasks:
        task_items = "".join(render_task_compact(r) for r in recent_tasks)
    else:
        task_items = '<div class="text-sm text-[#a1a1aa] py-4 text-center">No active tasks</div>'

    tasks_widget = f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-medium text-[#fafafa]">Active Tasks</h2>
    <a hx-get="/ui/tasks" hx-target="#tab-content" hx-push-url="true"
       class="text-xs text-[#71717a] cursor-pointer hover:text-[#fafafa] transition-colors">View all &rarr;</a>
  </div>
  {task_items}
</div>"""

    # Agents widget
    if active_agents:
        agent_items = []
        for row in active_agents:
            m = row._mapping
            dot_cls = "bg-[#22c55e] shadow-[0_0_4px_#22c55e]" if m["status"] == "running" else "bg-[#71717a]"
            proj = f'<span class="text-[#52525b]">{esc(m["project"])}</span>' if m["project"] else ""
            agent_items.append(
                f'<div class="flex items-center gap-3 py-2 border-b border-[#27272a] last:border-b-0">'
                f'<span class="w-2 h-2 rounded-full {dot_cls} shrink-0"></span>'
                f'<div class="flex-1 min-w-0">'
                f'<span class="text-sm font-medium">{esc(agent_display_name(m["username"]))}</span>'
                f' {proj}</div>'
                f'<div class="text-xs text-[#52525b] shrink-0">{time_ago(m["updated_at"])}</div>'
                f'</div>'
            )
        agents_html = "".join(agent_items)
    else:
        agents_html = '<div class="text-sm text-[#a1a1aa] py-4 text-center">No agents registered</div>'

    agents_widget = f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4">
  <h2 class="text-sm font-medium text-[#fafafa] mb-3">Agents</h2>
  {agents_html}
</div>"""

    # Journal widget
    if recent_journal:
        journal_items = "".join(render_journal_compact(r) for r in recent_journal)
    else:
        journal_items = '<div class="text-sm text-[#a1a1aa] py-4 text-center">No journal entries</div>'

    journal_widget = f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-medium text-[#fafafa]">Recent Journal</h2>
    <a hx-get="/ui/journal" hx-target="#tab-content" hx-push-url="true"
       class="text-xs text-[#71717a] cursor-pointer hover:text-[#fafafa] transition-colors">View all &rarr;</a>
  </div>
  {journal_items}
</div>"""

    # Recent runs widget
    from bioagentics.agent_api.database import runs as runs_table
    recent_runs = db.execute(
        select(runs_table).order_by(runs_table.c.started_at.desc()).limit(5)
    ).fetchall()

    if recent_runs:
        run_items = []
        for row in recent_runs:
            m = row._mapping
            is_running = m["finished_at"] is None
            dot = '<span class="w-2 h-2 rounded-full bg-[#22c55e] shadow-[0_0_4px_#22c55e] shrink-0"></span>' if is_running else '<span class="w-2 h-2 rounded-full bg-[#71717a] shrink-0"></span>'
            if m["exit_code"] is not None and m["exit_code"] != 0:
                dot = '<span class="w-2 h-2 rounded-full bg-[#ef4444] shrink-0"></span>'
            proj = f'<span class="text-[#52525b]">{esc(m["project"])}</span>' if m["project"] else ""
            cost = f'<span class="text-[#4ade80] text-xs">${float(m["cost_usd"]):.4f}</span>' if m["cost_usd"] else ""
            dur = _format_duration(m["duration_seconds"])
            run_items.append(
                f'<div class="flex items-center gap-3 py-2 border-b border-[#27272a] last:border-b-0">'
                f'{dot}'
                f'<span class="text-sm font-medium">{esc(m["agent"])}</span>'
                f'{proj}'
                f'<span class="text-xs text-[#52525b]">{esc(m["model"])}</span>'
                f'<span class="ml-auto flex items-center gap-3">'
                f'<span class="text-xs text-[#52525b] tabular-nums">{dur}</span>'
                f'{cost}'
                f'<span class="text-xs text-[#52525b]">{time_ago(m["started_at"])}</span>'
                f'</span></div>'
            )
        runs_html = "".join(run_items)
    else:
        runs_html = '<div class="text-sm text-[#a1a1aa] py-4 text-center">No runs yet</div>'

    runs_widget = f"""<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-4">
  <div class="flex justify-between items-center mb-3">
    <h2 class="text-sm font-medium text-[#fafafa]">Recent Runs</h2>
    <a hx-get="/ui/runs" hx-target="#tab-content" hx-push-url="true"
       class="text-xs text-[#71717a] cursor-pointer hover:text-[#fafafa] transition-colors">View all &rarr;</a>
  </div>
  {runs_html}
</div>"""

    return f"""<div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
  {human_banner}
  {tasks_widget}
  {agents_widget}
  {journal_widget}
  {runs_widget}
</div>"""


def render_projects_tab(db: Session, search: str, status: str, labels: str, offset: int, division: str = "") -> str:
    query = select(projects_table)
    count_query = select(func.count()).select_from(projects_table)

    if division:
        query = query.where(projects_table.c.division == division)
        count_query = count_query.where(projects_table.c.division == division)
    if status:
        query = query.where(projects_table.c.status == status)
        count_query = count_query.where(projects_table.c.status == status)
    if labels:
        query = query.where(projects_table.c.labels.contains(labels))
        count_query = count_query.where(projects_table.c.labels.contains(labels))
    if search:
        cond = projects_table.c.name.contains(search) | projects_table.c.description.contains(search)
        query = query.where(cond)
        count_query = count_query.where(cond)

    total = db.execute(count_query).scalar() or 0
    rows = db.execute(
        query.order_by(projects_table.c.updated_at.desc()).limit(PER_PAGE).offset(offset)
    ).fetchall()

    project_names = [r._mapping["name"] for r in rows]

    task_counts_map: dict[str, dict[str, int]] = {}
    if project_names:
        tc_rows = db.execute(
            select(tasks.c.project, tasks.c.status, func.count())
            .where(tasks.c.project.in_(project_names))
            .group_by(tasks.c.project, tasks.c.status)
        ).all()
        for proj, st, cnt in tc_rows:
            task_counts_map.setdefault(proj, {})[st] = cnt

    journal_counts_map: dict[str, int] = {}
    if project_names:
        jc_rows = db.execute(
            select(journal.c.project, func.count())
            .where(journal.c.project.in_(project_names))
            .group_by(journal.c.project)
        ).all()
        journal_counts_map = {proj: cnt for proj, cnt in jc_rows}

    path = "/ui/projects"
    status_options = [
        ("", "all statuses"),
        ("proposed", "proposed"),
        ("planning", "planning"),
        ("development", "development"),
        ("analysis", "analysis"),
        ("validation", "validation"),
        ("documentation", "documentation"),
        ("published", "published"),
        ("cancelled", "cancelled"),
    ]
    label_options = [
        ("", "all labels"),
        ("drug-candidate", "drug-candidate"),
        ("novel-finding", "novel-finding"),
        ("biomarker", "biomarker"),
        ("genomic", "genomic"),
        ("clinical", "clinical"),
        ("high-priority", "high-priority"),
        ("promising", "promising"),
    ]

    has_filters = bool(search or status or labels)
    filters_html = f"""<form class="flex gap-2 mb-4 flex-wrap items-center">
  {_search_input("search", search, "search research...", path)}
  {_select("status", status, status_options, path)}
  {_select("labels", labels, label_options, path)}
  {_clear_filters_btn(path, has_filters)}
</form>"""

    if rows:
        table_rows = "".join(
            render_project_row(r, task_counts_map.get(r._mapping["name"], {}), journal_counts_map.get(r._mapping["name"], 0))
            for r in rows
        )
        list_html = f"""<div class="overflow-x-auto">
  <table class="w-full min-w-[600px] border-collapse text-sm">
    <thead><tr>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Project</th>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Labels</th>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Status</th>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Tasks</th>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Journal</th>
      <th class="text-left px-2.5 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Updated</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>"""
    else:
        list_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">No projects found</div><div class="text-xs mt-1">Try adjusting your filters</div></div>'

    params = dict(search=search, status=status, labels=labels, division=division)
    pager_html = render_pager("p", path, offset, total, **params)

    return f"""<div class="flex items-center justify-between mb-4">
  <h2 class="text-lg font-semibold">Research Initiatives</h2>
  <span class="text-xs text-[#52525b] tabular-nums">{total} {"item" if total == 1 else "items"}</span>
</div>
{filters_html}
{list_html}
{pager_html}"""


def render_tasks_tab(
    db: Session,
    search: str,
    username: str,
    project: str,
    status: str,
    priority: str,
    sort: str,
    offset: int,
    division: str = "",
) -> str:
    # Fetch filter options for datalists
    usernames = sorted(
        db.execute(
            union(select(agents.c.username), select(tasks.c.username))
        ).scalars().all()
    )
    projects_list = sorted(
        db.execute(
            union(
                select(tasks.c.project).where(tasks.c.project.isnot(None)),
                select(agents.c.project).where(agents.c.project.isnot(None)),
            )
        ).scalars().all()
    )

    # Check for direct task ID lookup
    id_match = None
    if search and search.startswith("#") and search[1:].isdigit():
        id_match = int(search[1:])

    if id_match is not None:
        row = db.execute(select(tasks).where(tasks.c.id == id_match)).first()
        if row:
            items_html = f'<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{render_task_card(row)}</div>'
            total = 1
        else:
            items_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">Task {esc(search)} not found</div></div>'
            total = 0
    else:
        query = select(tasks)
        count_query = select(func.count()).select_from(tasks)

        if division:
            query = query.where(tasks.c.division == division)
            count_query = count_query.where(tasks.c.division == division)
        if username:
            query = query.where(tasks.c.username == username)
            count_query = count_query.where(tasks.c.username == username)
        if project:
            query = query.where(tasks.c.project == project)
            count_query = count_query.where(tasks.c.project == project)
        if status:
            query = query.where(tasks.c.status == status)
            count_query = count_query.where(tasks.c.status == status)
        if priority and priority.isdigit():
            query = query.where(tasks.c.priority == int(priority))
            count_query = count_query.where(tasks.c.priority == int(priority))
        if search:
            cond = tasks.c.title.contains(search) | tasks.c.description.contains(search)
            query = query.where(cond)
            count_query = count_query.where(cond)

        order = tasks.c.created_at.asc() if sort == "asc" else tasks.c.created_at.desc()
        total = db.execute(count_query).scalar() or 0
        rows = db.execute(query.order_by(order).limit(PER_PAGE).offset(offset)).fetchall()

        if rows:
            cards = "".join(render_task_card(r) for r in rows)
            items_html = f'<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{cards}</div>'
        else:
            items_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">No tasks</div><div class="text-xs mt-1">Create a task or adjust your filters</div></div>'

    path = "/ui/tasks"
    status_options = [
        ("", "all statuses"),
        ("pending", "pending"),
        ("in_progress", "in_progress"),
        ("blocked", "blocked"),
        ("done", "done"),
        ("cancelled", "cancelled"),
    ]
    priority_options = [
        ("", "all priorities"),
        ("5", "P5 (highest)"),
        ("4", "P4"),
        ("3", "P3"),
        ("2", "P2"),
        ("1", "P1 (lowest)"),
    ]
    sort_options = [("desc", "Newest first"), ("asc", "Oldest first")]

    dl_users = "".join(f'<option value="{esc(u)}">' for u in usernames)
    dl_projects = "".join(f'<option value="{esc(p)}">' for p in projects_list)

    has_filters = bool(search or username or project or status or priority)
    filters_html = f"""<form class="flex gap-2 mb-4 flex-wrap items-center">
  {_search_input("search", search, "search or #id...", path)}
  {_text_input("username", username, "agent", path, "dl-users")}
  {_text_input("project", project, "project", path, "dl-projects")}
  {_select("status", status, status_options, path)}
  {_select("priority", priority, priority_options, path)}
  {_select("sort", sort, sort_options, path)}
  <button type="button" class="bg-[#e6edf3] text-[#0d1117] border-none rounded-md px-3.5 py-2 text-sm font-semibold cursor-pointer hover:bg-[#cdd5de] min-h-[36px] transition-colors"
          onclick="openNewTaskModal()">+ New Task</button>
  {_clear_filters_btn(path, has_filters)}
</form>
<datalist id="dl-users">{dl_users}</datalist>
<datalist id="dl-projects">{dl_projects}</datalist>"""

    params = dict(search=search, username=username, project=project, status=status, priority=priority, sort=sort, division=division)
    pager_html = render_pager("t", path, offset, total, **params)

    return f"""<div class="flex items-center justify-between mb-4">
  <h2 class="text-lg font-semibold">Tasks</h2>
  <span class="text-xs text-[#52525b] tabular-nums">{total} {"item" if total == 1 else "items"}</span>
</div>
{filters_html}
{items_html}
{pager_html}"""


def render_journal_tab(
    db: Session,
    search: str,
    username: str,
    project: str,
    sort: str,
    offset: int,
    division: str = "",
) -> str:
    usernames = sorted(
        db.execute(
            union(select(agents.c.username), select(journal.c.username))
        ).scalars().all()
    )
    projects_list = sorted(
        db.execute(
            union(
                select(journal.c.project).where(journal.c.project.isnot(None)),
                select(agents.c.project).where(agents.c.project.isnot(None)),
            )
        ).scalars().all()
    )

    query = select(journal)
    count_query = select(func.count()).select_from(journal)

    if division:
        query = query.where(journal.c.division == division)
        count_query = count_query.where(journal.c.division == division)
    if username:
        query = query.where(journal.c.username == username)
        count_query = count_query.where(journal.c.username == username)
    if project:
        query = query.where(journal.c.project == project)
        count_query = count_query.where(journal.c.project == project)
    if search:
        query = query.where(journal.c.content.contains(search))
        count_query = count_query.where(journal.c.content.contains(search))

    order = journal.c.created_at.asc() if sort == "asc" else journal.c.created_at.desc()
    total = db.execute(count_query).scalar() or 0
    rows = db.execute(query.order_by(order).limit(PER_PAGE).offset(offset)).fetchall()

    if rows:
        cards = "".join(render_journal_card(r) for r in rows)
        items_html = f'<div class="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">{cards}</div>'
    else:
        items_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">No journal entries</div><div class="text-xs mt-1">Entries will appear as agents log their activity</div></div>'

    path = "/ui/journal"
    sort_options = [("desc", "Newest first"), ("asc", "Oldest first")]

    dl_users = "".join(f'<option value="{esc(u)}">' for u in usernames)
    dl_projects = "".join(f'<option value="{esc(p)}">' for p in projects_list)

    has_filters = bool(search or username or project)
    filters_html = f"""<form class="flex gap-2 mb-4 flex-wrap items-center">
  {_search_input("search", search, "search content...", path)}
  {_text_input("username", username, "agent", path, "dl-users")}
  {_text_input("project", project, "project", path, "dl-projects")}
  {_select("sort", sort, sort_options, path)}
  {_clear_filters_btn(path, has_filters)}
</form>
<datalist id="dl-users">{dl_users}</datalist>
<datalist id="dl-projects">{dl_projects}</datalist>"""

    params = dict(search=search, username=username, project=project, sort=sort, division=division)
    pager_html = render_pager("j", path, offset, total, **params)

    return f"""<div class="flex items-center justify-between mb-4">
  <h2 class="text-lg font-semibold">Journal</h2>
  <span class="text-xs text-[#52525b] tabular-nums">{total} {"item" if total == 1 else "items"}</span>
</div>
{filters_html}
{items_html}
{pager_html}"""


def _format_duration(seconds) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _format_tokens(n) -> str:
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _cache_badge(m) -> str:
    cache_read = m.get("cache_read_tokens") or 0
    input_tok = m.get("input_tokens") or 0
    if not cache_read or not input_tok:
        return ""
    pct = cache_read / input_tok * 100
    return f'<div class="text-[10px] text-[#52525b]">cache {pct:.0f}%</div>'


def render_run_row(row) -> str:
    m = row._mapping
    is_running = m["finished_at"] is None

    status_html = (
        '<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#052e16] text-[#4ade80]">running</span>'
        if is_running
        else '<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#27272a] text-[#a1a1aa]">finished</span>'
    )
    if m["exit_code"] is not None and m["exit_code"] != 0:
        status_html = f'<span class="inline-flex items-center px-2 py-0.5 rounded-full text-[11px] font-medium bg-[#450a0a] text-[#f87171]">exit {m["exit_code"]}</span>'

    project_html = ""
    if m["project"]:
        pq = build_qs(project=m["project"])
        project_html = f'<a class="text-[#3b82f6] text-xs no-underline hover:underline cursor-pointer" hx-get="/ui/runs?{pq}" hx-target="#tab-content" hx-push-url="true">{esc(m["project"])}</a>'

    cost_html = ""
    if m["cost_usd"]:
        cost_html = f'<span class="text-[#4ade80]">${float(m["cost_usd"]):.4f}</span>'

    return f"""<tr class="hover:bg-[#18181b] transition-colors">
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs font-mono text-[#71717a]">#{m['id']}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a]">
    <div class="text-sm font-medium">{esc(m['agent'])}</div>
    {project_html}
  </td>
  <td class="px-3 py-2.5 border-b border-[#27272a]">{status_html}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs text-[#a1a1aa]">{esc(m['backend'])}<span class="text-[#52525b]"> / </span>{esc(m['model'])}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs tabular-nums text-[#a1a1aa]">{_format_duration(m['duration_seconds'])}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs tabular-nums text-[#a1a1aa]">{_format_tokens(m['input_tokens'])}<span class="text-[#52525b]"> / </span>{_format_tokens(m['output_tokens'])}{_cache_badge(m)}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs tabular-nums">{cost_html}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] text-xs tabular-nums text-[#a1a1aa]">{m['tasks_completed']}</td>
  <td class="px-3 py-2.5 border-b border-[#27272a] whitespace-nowrap">{time_tag(m['started_at'])}</td>
</tr>"""


def render_runs_tab(
    db: Session,
    agent: str,
    project: str,
    sort: str,
    offset: int,
    division: str = "",
) -> str:
    from bioagentics.agent_api.database import runs

    query = select(runs)
    count_query = select(func.count()).select_from(runs)

    if division:
        query = query.where(runs.c.division == division)
        count_query = count_query.where(runs.c.division == division)
    if agent:
        query = query.where(runs.c.agent == agent)
        count_query = count_query.where(runs.c.agent == agent)
    if project:
        query = query.where(runs.c.project == project)
        count_query = count_query.where(runs.c.project == project)

    order = runs.c.started_at.asc() if sort == "asc" else runs.c.started_at.desc()
    total = db.execute(count_query).scalar() or 0
    rows = db.execute(query.order_by(order).limit(PER_PAGE).offset(offset)).fetchall()

    # Get filter options
    agent_names = sorted(
        db.execute(select(runs.c.agent.distinct()).where(runs.c.agent.isnot(None))).scalars().all()
    )
    project_names = sorted(
        db.execute(select(runs.c.project.distinct()).where(runs.c.project.isnot(None))).scalars().all()
    )

    path = "/ui/runs"
    sort_options = [("desc", "Newest first"), ("asc", "Oldest first")]

    dl_agents = "".join(f'<option value="{esc(a)}">' for a in agent_names)
    dl_projects = "".join(f'<option value="{esc(p)}">' for p in project_names)

    has_filters = bool(agent or project)
    filters_html = f"""<form class="flex gap-2 mb-4 flex-wrap items-center">
  {_text_input("agent", agent, "agent", path, "dl-run-agents")}
  {_text_input("project", project, "project", path, "dl-run-projects")}
  {_select("sort", sort, sort_options, path)}
  {_clear_filters_btn(path, has_filters)}
</form>
<datalist id="dl-run-agents">{dl_agents}</datalist>
<datalist id="dl-run-projects">{dl_projects}</datalist>"""

    if rows:
        table_rows = "".join(render_run_row(r) for r in rows)
        list_html = f"""<div class="overflow-x-auto">
  <table class="w-full min-w-[700px] border-collapse text-sm">
    <thead><tr>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">ID</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Agent</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Status</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Backend / Model</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Duration</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Tokens (in/out)</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Cost</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Tasks</th>
      <th class="text-left px-3 py-2 border-b-2 border-[#27272a] text-[#71717a] text-xs font-medium whitespace-nowrap">Started</th>
    </tr></thead>
    <tbody>{table_rows}</tbody>
  </table>
</div>"""
    else:
        list_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">No runs recorded</div><div class="text-xs mt-1">Runs will appear as agents start executing</div></div>'

    params = dict(agent=agent, project=project, sort=sort, division=division)
    pager_html = render_pager("r", path, offset, total, **params)

    return f"""<div class="flex items-center justify-between mb-4">
  <h2 class="text-lg font-semibold">Runs</h2>
  <span class="text-xs text-[#52525b] tabular-nums">{total} {"run" if total == 1 else "runs"}</span>
</div>
{filters_html}
{list_html}
{pager_html}"""


def render_keys_tab(db: Session, api_key: str) -> str:
    authed = check_api_key(db, api_key)

    btn = """<div class="flex gap-2 mb-4 flex-wrap items-center">
  <button type="button" class="bg-[#e6edf3] text-[#0d1117] border-none rounded-md px-3.5 py-2 text-sm font-semibold cursor-pointer hover:bg-[#cdd5de] min-h-[36px] transition-colors"
          onclick="openNewKeyModal()">+ New Key</button>
</div>"""

    if not authed:
        return f"""{btn}
<div id="key-banner"></div>
<div class="text-center text-[#a1a1aa] py-10 text-sm">Enter a valid API key in the sidebar to manage keys.</div>"""

    rows = db.execute(
        select(api_keys).order_by(api_keys.c.created_at.desc())
    ).fetchall()

    # Mask keys - show only first 8 chars
    class MaskedRow:
        def __init__(self, row):
            m = dict(row._mapping)
            m["key"] = m["key"][:8] + "..."
            self._mapping = m

    total = len(rows)
    if rows:
        items_html = '<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg overflow-hidden shadow-sm">'
        items_html += "".join(render_key_row(MaskedRow(r)) for r in rows)
        items_html += "</div>"
    else:
        items_html = f'<div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">No API keys</div><div class="text-xs mt-1">Auth is currently disabled (open access)</div></div>'

    return f"""<div class="flex items-center justify-between mb-4">
  <h2 class="text-lg font-semibold">API Keys</h2>
  <span class="text-xs text-[#52525b] tabular-nums">{total} {"key" if total == 1 else "keys"}</span>
</div>
{btn}
<div id="key-banner"></div>
{items_html}"""


# ── Stats & Presence Partials ─────────────────────────────────────────


def render_stats_html(db: Session, division: str = "") -> str:
    task_q = select(tasks.c.status, func.count()).group_by(tasks.c.status)
    journal_q = select(func.count()).select_from(journal)
    if division:
        task_q = task_q.where(tasks.c.division == division)
        journal_q = journal_q.where(journal.c.division == division)
    task_rows = db.execute(task_q).all()
    task_counts = {row[0]: row[1] for row in task_rows}
    total_tasks = sum(task_counts.values())
    running_agents = db.execute(
        select(func.count(agents.c.username.distinct())).where(agents.c.status == "running")
    ).scalar() or 0
    journal_count = db.execute(journal_q).scalar() or 0

    pending = task_counts.get("pending", 0)
    in_progress = task_counts.get("in_progress", 0)
    blocked = task_counts.get("blocked", 0)
    done = task_counts.get("done", 0)
    cancelled = task_counts.get("cancelled", 0)

    done_pct = int(done / total_tasks * 100) if total_tasks > 0 else 0

    return f"""<div class="grid grid-cols-2 lg:grid-cols-4 gap-3">
  <div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg px-4 py-3 cursor-pointer hover:border-[#3f3f46] transition-colors"
       hx-get="/ui/tasks" hx-target="#tab-content" hx-push-url="true">
    <div class="text-xs text-[#71717a] font-medium mb-1">Tasks</div>
    <div class="text-2xl font-semibold tabular-nums">{total_tasks}</div>
    <div class="mt-2 w-full bg-[#18181b] rounded-full h-1 overflow-hidden">
      <div class="h-full bg-[#3b82f6] rounded-full transition-all" style="width:{done_pct}%"></div>
    </div>
    <div class="flex gap-2 mt-1.5 text-[11px] text-[#52525b] tabular-nums">
      <span>{pending}p</span><span>{in_progress}a</span><span>{blocked}b</span><span>{done}d</span>
    </div>
  </div>
  <div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg px-4 py-3 cursor-pointer hover:border-[#3f3f46] transition-colors"
       hx-get="/ui/tasks?status=done" hx-target="#tab-content" hx-push-url="true">
    <div class="text-xs text-[#71717a] font-medium mb-1">Completed</div>
    <div class="text-2xl font-semibold tabular-nums">{done}</div>
    <div class="text-[11px] text-[#52525b] mt-1.5 tabular-nums">{cancelled} cancelled</div>
  </div>
  <div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg px-4 py-3 cursor-pointer hover:border-[#3f3f46] transition-colors"
       hx-get="/ui/dashboard" hx-target="#tab-content" hx-push-url="true">
    <div class="text-xs text-[#71717a] font-medium mb-1">Agents</div>
    <div class="text-2xl font-semibold tabular-nums">{running_agents}</div>
    <div class="text-[11px] text-[#52525b] mt-1.5">running</div>
  </div>
  <div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg px-4 py-3 cursor-pointer hover:border-[#3f3f46] transition-colors"
       hx-get="/ui/journal" hx-target="#tab-content" hx-push-url="true">
    <div class="text-xs text-[#71717a] font-medium mb-1">Journal</div>
    <div class="text-2xl font-semibold tabular-nums">{journal_count}</div>
  </div>
</div>"""


def render_presence_html(db: Session) -> str:
    rows = db.execute(
        select(agents).order_by(agents.c.updated_at.desc())
    ).fetchall()

    if not rows:
        return '<div class="text-xs text-[#a1a1aa] italic py-1">no agents online</div>'

    items = []
    for row in rows:
        m = row._mapping
        dot_cls = "bg-[#22c55e] shadow-[0_0_4px_#22c55e]" if m["status"] == "running" else "bg-[#71717a]"
        project_html = ""
        if m["project"]:
            project_html = f'<div class="text-[11px] text-[#a1a1aa] truncate">{esc(m["project"])}</div>'
        items.append(
            f'<div class="flex items-center gap-2 py-1.5">'
            f'<span class="w-2 h-2 rounded-full shrink-0 {dot_cls}"></span>'
            f'<div class="min-w-0 flex-1">'
            f'<div class="text-xs font-medium truncate">{esc(agent_display_name(m["username"]))}</div>'
            f'{project_html}'
            f'</div>'
            f'</div>'
        )
    return "".join(items)


# ── Sidebar Navigation ───────────────────────────────────────────────


def render_sidebar_nav(active: str, human_count: int = 0) -> str:
    tabs_def = [
        ("dashboard", "Dashboard", ICONS["dashboard"]),
        ("projects", "Research", ICONS["folder"]),
        ("tasks", "Tasks", ICONS["tasks"]),
        ("journal", "Lab Notebook", ICONS["journal"]),
        ("runs", "Runs", ICONS["play"]),
        ("keys", "Keys", ICONS["key"]),
    ]
    items = []
    for tab_id, label, icon in tabs_def:
        if tab_id == active:
            active_cls = "bg-[#18181b] text-[#fafafa] border-l-2 border-l-[#3b82f6]"
        else:
            active_cls = "text-[#71717a] hover:text-[#fafafa] hover:bg-[#18181b] border-l-2 border-l-transparent"
        badge = ""
        if tab_id == "tasks" and human_count > 0:
            badge = f'<span class="ml-auto bg-[#ef4444] text-white text-[10px] font-medium rounded-full min-w-[18px] h-[18px] flex items-center justify-center px-1">{human_count}</span>'
        items.append(
            f'<a class="flex items-center gap-2.5 px-3 py-2 rounded-md text-sm font-medium {active_cls}'
            f' transition-colors cursor-pointer no-underline"'
            f' hx-get="/ui/{tab_id}" hx-target="#tab-content" hx-push-url="true">'
            f'{icon}<span class="flex-1">{label}</span>{badge}</a>'
        )
    return "\n    ".join(items)


def render_sidebar_nav_oob(active: str, db: Session | None = None) -> str:
    human_count = _human_task_count(db) if db else 0
    return f'<nav id="sidebar-nav" class="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto" hx-swap-oob="innerHTML">{render_sidebar_nav(active, human_count)}</nav>'


# ── Shell & Landing Page ─────────────────────────────────────────────

CLIENT_JS = """
function getApiKey() { return localStorage.getItem('agent_comms_api_key') || ''; }
function saveApiKey() {
  var key = document.getElementById('api-key').value.trim();
  if (key) { localStorage.setItem('agent_comms_api_key', key); }
  else { localStorage.removeItem('agent_comms_api_key'); }
  var st = document.getElementById('api-key-status');
  st.textContent = key ? 'Saved' : 'Cleared';
  setTimeout(function() { st.textContent = ''; }, 2000);
}
document.addEventListener('DOMContentLoaded', function() {
  var saved = getApiKey();
  if (saved) { document.getElementById('api-key').value = saved; }
});

// Division filter
function getDivision() { return localStorage.getItem('bioagentics_division') || ''; }
function setDivision(val) {
  if (val) { localStorage.setItem('bioagentics_division', val); }
  else { localStorage.removeItem('bioagentics_division'); }
  // Reload current tab content
  var tabContent = document.getElementById('tab-content');
  if (tabContent) {
    var url = new URL(window.location.href);
    if (val) { url.searchParams.set('division', val); } else { url.searchParams.delete('division'); }
    htmx.ajax('GET', url.pathname + '?' + url.searchParams.toString(), {target: '#tab-content', pushUrl: true});
  }
}
document.addEventListener('DOMContentLoaded', function() {
  var saved = getDivision();
  var sel = document.getElementById('division-filter');
  if (sel && saved) { sel.value = saved; }
});

// Inject API key + division on all htmx requests
document.addEventListener('htmx:configRequest', function(evt) {
  var key = getApiKey();
  if (key) { evt.detail.headers['X-API-Key'] = key; }
  var div = getDivision();
  if (div && !evt.detail.parameters['division']) { evt.detail.parameters['division'] = div; }
});

// Sidebar toggle (mobile)
function toggleSidebar() {
  var sb = document.getElementById('sidebar');
  var ov = document.getElementById('sidebar-overlay');
  if (sb.classList.contains('-translate-x-full')) {
    sb.classList.remove('-translate-x-full');
    ov.classList.remove('hidden');
  } else {
    sb.classList.add('-translate-x-full');
    ov.classList.add('hidden');
  }
}
// Restore focus to the active input after htmx swaps (prevents search losing focus)
var _pendingFocus = null;
document.addEventListener('htmx:beforeSwap', function(evt) {
  var ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA') && ae.name && evt.detail.target.id === 'tab-content') {
    _pendingFocus = { name: ae.name, pos: ae.selectionStart, end: ae.selectionEnd };
  } else {
    _pendingFocus = null;
  }
});
document.addEventListener('htmx:afterSettle', function(evt) {
  if (_pendingFocus && evt.detail.target.id === 'tab-content') {
    var el = document.querySelector('#tab-content [name="' + _pendingFocus.name + '"]');
    if (el) {
      el.focus();
      try { el.setSelectionRange(_pendingFocus.pos, _pendingFocus.end); } catch(e) {}
    }
    _pendingFocus = null;
  }
});
// Auto-close sidebar on user navigation (mobile), but not on auto-refresh
var _autoRefreshing = false;
document.addEventListener('htmx:afterSwap', function() {
  if (_autoRefreshing) return;
  if (window.innerWidth < 1024) {
    var sb = document.getElementById('sidebar');
    var ov = document.getElementById('sidebar-overlay');
    if (sb && !sb.classList.contains('-translate-x-full')) {
      sb.classList.add('-translate-x-full');
      if (ov) ov.classList.add('hidden');
    }
  }
});

// Task action modal
var actionTaskId = null, actionStatus = null;
function openActionModal(taskId, status) {
  actionTaskId = taskId;
  actionStatus = status;
  var isDone = status === 'done';
  var overlay = document.createElement('div');
  overlay.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-50 modal-enter';
  overlay.id = 'action-modal';
  overlay.innerHTML = '<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-5 w-[420px] max-w-[90vw]">' +
    '<h2 class="text-base font-semibold mb-1">' + (isDone ? 'Mark Task Done' : 'Reject Task') + '</h2>' +
    '<div class="text-sm text-[#a1a1aa] mb-3">' + (isDone ? 'Provide a brief summary of what was completed.' : 'Provide a reason for rejecting this task.') + '</div>' +
    '<textarea id="action-summary" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md p-2.5 text-sm font-[inherit] resize-y min-h-[80px] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors" placeholder="' + (isDone ? 'Summary of work done...' : 'Reason for rejection...') + '"></textarea>' +
    '<div class="flex gap-2 justify-end mt-3">' +
    '<button class="bg-[#18181b] text-[#fafafa] px-4 py-2 rounded-md text-sm font-medium cursor-pointer border border-[#3f3f46] min-h-[36px] hover:bg-[#27272a] transition-colors" onclick="closeActionModal()">Cancel</button>' +
    '<button class="' + (isDone ? 'bg-[#22c55e] text-black' : 'bg-[#ef4444] text-white') + ' px-4 py-2 rounded-md text-sm font-medium cursor-pointer border-none min-h-[36px]" id="action-submit" onclick="submitAction()">' + (isDone ? 'Complete' : 'Reject') + '</button>' +
    '</div></div>';
  document.body.appendChild(overlay);
  overlay.addEventListener('click', function(e) { if (e.target === overlay) closeActionModal(); });
  document.getElementById('action-summary').focus();
}
function closeActionModal() {
  var m = document.getElementById('action-modal');
  if (m) m.remove();
  actionTaskId = null; actionStatus = null;
}
async function submitAction() {
  var summary = document.getElementById('action-summary').value.trim();
  if (!summary) { document.getElementById('action-summary').style.borderColor = '#f85149'; return; }
  var btn = document.getElementById('action-submit');
  btn.textContent = '...'; btn.disabled = true;
  try {
    var res = await fetch('/api/tasks/' + actionTaskId, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': getApiKey() },
      body: JSON.stringify({ status: actionStatus, description: summary })
    });
    if (!res.ok) throw new Error('Failed');
    closeActionModal();
    htmx.ajax('GET', window.location.pathname + window.location.search, {target: '#tab-content', swap: 'innerHTML'});
  } catch (e) {
    btn.textContent = 'Retry'; btn.disabled = false;
    alert('Failed to update task: ' + e.message);
  }
}

// New task modal
function openNewTaskModal() {
  var overlay = document.createElement('div');
  overlay.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-50 modal-enter';
  overlay.id = 'new-task-modal';
  overlay.innerHTML = '<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-5 w-[420px] max-w-[90vw]">' +
    '<h2 class="text-base font-semibold mb-1">New Task</h2>' +
    '<div class="text-sm text-[#a1a1aa] mb-3">Assign a task to an agent or team member.</div>' +
    '<label class="block text-xs text-[#a1a1aa] mb-1">Assign to</label>' +
    '<input id="nt-user" list="dl-users" placeholder="agent" autocomplete="off" data-1p-ignore data-lpignore="true" data-form-type="other" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md p-2.5 text-sm mb-2 min-h-[36px] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors" />' +
    '<label class="block text-xs text-[#a1a1aa] mb-1">Title</label>' +
    '<input id="nt-title" placeholder="Task title" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md p-2.5 text-sm mb-2 min-h-[36px] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors" />' +
    '<div class="flex gap-2.5">' +
    '<div class="flex-1"><label class="block text-xs text-[#a1a1aa] mb-1">Priority</label>' +
    '<div class="relative">' +
    '<select id="nt-priority" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md pl-3 pr-8 py-2.5 text-sm min-h-[36px] appearance-none cursor-pointer focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors">' +
    '<option value="1">P1 (lowest)</option><option value="2">P2</option><option value="3" selected>P3</option><option value="4">P4</option><option value="5">P5 (highest)</option></select>' +
    '<span class="absolute right-2 top-1/2 -translate-y-1/2 text-[#52525b] pointer-events-none"><svg class="w-4 h-4" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M6 9l6 6 6-6"/></svg></span></div></div>' +
    '<div class="flex-1"><label class="block text-xs text-[#a1a1aa] mb-1">Project</label>' +
    '<input id="nt-project" list="dl-projects" placeholder="optional" autocomplete="off" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md p-2.5 text-sm min-h-[36px] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors" /></div></div>' +
    '<label class="block text-xs text-[#a1a1aa] mb-1 mt-2.5">Description</label>' +
    '<textarea id="nt-desc" placeholder="Describe what needs to be done..." class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-lg p-2 text-sm font-[inherit] resize-y min-h-[80px]"></textarea>' +
    '<div class="flex gap-2 justify-end mt-3">' +
    '<button class="bg-[#18181b] text-[#fafafa] px-4 py-2 rounded-md text-sm font-medium cursor-pointer border border-[#3f3f46] min-h-[36px] hover:bg-[#27272a] transition-colors" onclick="closeNewTaskModal()">Cancel</button>' +
    '<button class="bg-[#22c55e] text-black px-4 py-2 rounded-md text-sm font-medium cursor-pointer border-none min-h-[36px]" id="nt-submit" onclick="submitNewTask()">Create Task</button>' +
    '</div></div>';
  document.body.appendChild(overlay);
  overlay.addEventListener('click', function(e) { if (e.target === overlay) closeNewTaskModal(); });
  document.getElementById('nt-user').focus();
}
function closeNewTaskModal() {
  var m = document.getElementById('new-task-modal');
  if (m) m.remove();
}
async function submitNewTask() {
  var user = document.getElementById('nt-user').value.trim();
  var title = document.getElementById('nt-title').value.trim();
  if (!user) { document.getElementById('nt-user').style.borderColor = '#f85149'; return; }
  if (!title) { document.getElementById('nt-title').style.borderColor = '#f85149'; return; }
  var btn = document.getElementById('nt-submit');
  btn.textContent = '...'; btn.disabled = true;
  var body = { username: user, title: title, priority: parseInt(document.getElementById('nt-priority').value) };
  var div = getDivision();
  if (div) body.division = div;
  var project = document.getElementById('nt-project').value.trim();
  if (project) body.project = project;
  var desc = document.getElementById('nt-desc').value.trim();
  if (desc) body.description = desc;
  try {
    var res = await fetch('/api/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': getApiKey() },
      body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error('Failed');
    closeNewTaskModal();
    htmx.ajax('GET', window.location.pathname + window.location.search, {target: '#tab-content', swap: 'innerHTML'});
    htmx.ajax('GET', '/ui/partials/stats', {target: '#stats-bar', swap: 'innerHTML'});
  } catch (e) {
    btn.textContent = 'Retry'; btn.disabled = false;
    alert('Failed to create task: ' + e.message);
  }
}

// Keys management
function openNewKeyModal() {
  var overlay = document.createElement('div');
  overlay.className = 'fixed inset-0 bg-black/60 flex items-center justify-center z-50 modal-enter';
  overlay.id = 'new-key-modal';
  overlay.innerHTML = '<div class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-5 w-[420px] max-w-[90vw]">' +
    '<h2 class="text-base font-semibold mb-1">Create API Key</h2>' +
    '<div class="text-sm text-[#a1a1aa] mb-3">Give this key a name to identify what system or agent uses it.</div>' +
    '<label class="block text-xs text-[#a1a1aa] mb-1">Name</label>' +
    '<input id="nk-name" placeholder="e.g. ci-pipeline, agent-alpha, matt" autocomplete="off" class="w-full bg-[#09090b] border border-[#27272a] text-[#fafafa] rounded-md p-2.5 text-sm min-h-[36px] focus:outline-none focus:ring-2 focus:ring-[#3b82f6]/40 focus:border-[#3b82f6] transition-colors" />' +
    '<div class="flex gap-2 justify-end mt-3">' +
    '<button class="bg-[#18181b] text-[#fafafa] px-4 py-2 rounded-md text-sm font-medium cursor-pointer border border-[#3f3f46] min-h-[36px] hover:bg-[#27272a] transition-colors" onclick="closeNewKeyModal()">Cancel</button>' +
    '<button class="bg-[#22c55e] text-black px-4 py-2 rounded-md text-sm font-medium cursor-pointer border-none min-h-[36px]" id="nk-submit" onclick="submitNewKey()">Create</button>' +
    '</div></div>';
  document.body.appendChild(overlay);
  overlay.addEventListener('click', function(e) { if (e.target === overlay) closeNewKeyModal(); });
  document.getElementById('nk-name').focus();
}
function closeNewKeyModal() {
  var m = document.getElementById('new-key-modal');
  if (m) m.remove();
}
async function submitNewKey() {
  var name = document.getElementById('nk-name').value.trim();
  if (!name) { document.getElementById('nk-name').style.borderColor = '#f85149'; return; }
  var btn = document.getElementById('nk-submit');
  btn.textContent = '...'; btn.disabled = true;
  try {
    var res = await fetch('/api/keys', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': getApiKey() },
      body: JSON.stringify({ name: name })
    });
    if (res.status === 401) { btn.textContent = 'Create'; btn.disabled = false; alert('Invalid API key. Enter a valid key to create new keys.'); return; }
    if (!res.ok) throw new Error('Failed');
    var key = await res.json();
    closeNewKeyModal();
    var banner = document.getElementById('key-banner');
    if (banner) {
      banner.innerHTML = '<div class="bg-[#052e16] border border-[#166534] rounded-lg px-4 py-3 mb-3">' +
        '<div class="text-sm font-semibold mb-1">Key created for \\u201c' + name.replace(/[<>&"]/g, '') + '\\u201d</div>' +
        '<div class="font-mono text-sm text-[#4ade80] break-all select-all">' + key.key + '</div>' +
        '<div class="text-xs text-[#fbbf24] mt-1">Copy this key now \\u2014 it will not be shown again in full.</div></div>';
    }
    htmx.ajax('GET', '/ui/keys', {target: '#tab-content', swap: 'innerHTML'});
  } catch (e) {
    btn.textContent = 'Retry'; btn.disabled = false;
    alert('Failed to create key: ' + e.message);
  }
}
// Loading bar
(function() {
  var bar = null;
  document.addEventListener('htmx:beforeRequest', function() {
    bar = document.getElementById('loading-bar');
    if (bar) { bar.style.width = '70%'; bar.classList.add('active'); }
  });
  document.addEventListener('htmx:afterSettle', function() {
    bar = document.getElementById('loading-bar');
    if (bar) { bar.style.width = '100%'; setTimeout(function() { bar.classList.remove('active'); bar.style.width = '0'; }, 200); }
  });
  document.addEventListener('htmx:responseError', function() {
    bar = document.getElementById('loading-bar');
    if (bar) { bar.style.background = '#ef4444'; bar.style.width = '100%'; setTimeout(function() { bar.classList.remove('active'); bar.style.width = '0'; bar.style.background = '#3b82f6'; }, 1000); }
  });
})();

// Client-side relative timestamp updates
setInterval(function() {
  document.querySelectorAll('.timestamp[data-ts]').forEach(function(el) {
    var ts = el.dataset.ts;
    if (!ts) return;
    try {
      var dt = new Date(ts.replace('Z', '+00:00'));
      var diff = (Date.now() - dt.getTime()) / 1000;
      var txt;
      if (diff < 60) txt = 'just now';
      else if (diff < 3600) txt = Math.floor(diff / 60) + 'm ago';
      else if (diff < 86400) txt = Math.floor(diff / 3600) + 'h ago';
      else txt = Math.floor(diff / 86400) + 'd ago';
      el.textContent = txt;
    } catch(e) {}
  });
}, 60000);

// Auto-refresh tab content every 30s (skip if user is interacting)
setInterval(function() {
  var ae = document.activeElement;
  if (ae && (ae.tagName === 'INPUT' || ae.tagName === 'TEXTAREA' || ae.tagName === 'SELECT')) return;
  // Skip if sidebar is open on mobile
  var sb = document.getElementById('sidebar');
  if (sb && window.innerWidth < 1024 && !sb.classList.contains('-translate-x-full')) return;
  // Skip if a modal is open
  if (document.querySelector('#action-modal, #new-task-modal, #new-key-modal')) return;
  var tc = document.getElementById('tab-content');
  if (tc) {
    // Remember which <details> elements are open so we can restore after swap
    var openDetails = [];
    tc.querySelectorAll('details').forEach(function(d, i) { if (d.open) openDetails.push(i); });
    _autoRefreshing = true;
    var url = window.location.pathname + window.location.search;
    htmx.ajax('GET', url, {target: '#tab-content', swap: 'innerHTML'}).then(function() {
      _autoRefreshing = false;
      openDetails.forEach(function(i) { var d = tc.querySelectorAll('details')[i]; if (d) d.open = true; });
    }).catch(function() {
      _autoRefreshing = false;
    });
  }
}, 30000);

async function revokeKey(id, name) {
  if (!confirm('Revoke key "' + name + '"? Any system using it will lose write access.')) return;
  try {
    var res = await fetch('/api/keys/' + id, {
      method: 'DELETE',
      headers: { 'X-API-Key': getApiKey() }
    });
    if (res.status === 401) { alert('Invalid API key.'); return; }
    if (!res.ok && res.status !== 204) throw new Error('Failed');
    htmx.ajax('GET', '/ui/keys', {target: '#tab-content', swap: 'innerHTML'});
  } catch (e) {
    alert('Failed to revoke key: ' + e.message);
  }
}
"""


SHELL_CSS = """
  /* Loading bar */
  #loading-bar { position: fixed; top: 0; left: 0; height: 2px; background: #3b82f6; z-index: 9999; transition: width 300ms ease; width: 0; opacity: 0; }
  #loading-bar.active { opacity: 1; }

  #tab-content { transition: opacity 150ms ease-in-out; }
  .htmx-swapping > #tab-content { opacity: 0; }
  .modal-enter { animation: fadeIn 150ms ease-out; }
  @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
  .modal-enter > div { animation: slideUp 150ms ease-out; }
  @keyframes slideUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .line-clamp-2 { display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; }
  .line-clamp-3 { display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .line-clamp-6 { display: -webkit-box; -webkit-line-clamp: 6; -webkit-box-orient: vertical; overflow: hidden; }

  /* Timestamp tooltips */
  .timestamp { position: relative; }
  .timestamp:hover::after {
    content: attr(data-ts);
    position: absolute;
    bottom: calc(100% + 6px);
    left: 50%;
    transform: translateX(-50%);
    background: #18181b;
    border: 1px solid #3f3f46;
    color: #fafafa;
    font-size: 11px;
    font-family: monospace;
    padding: 4px 8px;
    border-radius: 6px;
    white-space: nowrap;
    z-index: 50;
    pointer-events: none;
    box-shadow: 0 4px 12px rgba(0,0,0,0.4);
  }
  .timestamp:hover::before {
    content: '';
    position: absolute;
    bottom: calc(100% + 2px);
    left: 50%;
    transform: translateX(-50%);
    border: 4px solid transparent;
    border-top-color: #18181b;
    z-index: 50;
    pointer-events: none;
  }
"""


def _human_task_count(db: Session) -> int:
    return db.execute(
        select(func.count()).select_from(tasks)
        .where(tasks.c.username == "human")
        .where(tasks.c.status.in_(["pending", "in_progress", "blocked"]))
    ).scalar() or 0


def render_shell(active_tab: str, content: str, stats_html: str, presence_html: str, human_count: int = 0) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent Comms</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='20' fill='%2309090b'/><circle cx='50' cy='50' r='20' fill='%233b82f6'/></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>{SHELL_CSS}</style>
</head>
<body class="bg-[#09090b] text-[#fafafa] font-['Inter',system-ui,sans-serif] leading-relaxed antialiased">

<!-- Sidebar -->
<aside id="sidebar" class="fixed top-0 left-0 h-screen w-60 bg-[#09090b] border-r border-[#27272a]
                           flex flex-col z-40 transition-transform duration-200
                           -translate-x-full lg:translate-x-0">
  <div class="px-4 py-4 border-b border-[#27272a]">
    <div class="flex items-center justify-between">
      <h1 class="text-sm font-semibold tracking-tight">BioAgentics</h1>
      <span class="text-[#52525b] text-xs">local</span>
    </div>
  </div>

  <div class="px-3 py-2.5 border-b border-[#27272a]">
    <div class="text-[10px] text-[#52525b] font-medium uppercase tracking-wider mb-1.5">Division</div>
    <select id="division-filter" onchange="setDivision(this.value)"
            class="w-full bg-[#18181b] border border-[#27272a] text-sm text-[#fafafa] px-2 py-1.5 rounded-md focus:outline-none focus:ring-1 focus:ring-[#3b82f6]/40 cursor-pointer">
      <option value="">All</option>
      <option value="cancer">Cancer</option>
      <option value="crohns">Crohn's</option>
    </select>
  </div>

  <nav id="sidebar-nav" class="flex-1 px-2 py-3 space-y-0.5 overflow-y-auto">
    {render_sidebar_nav(active_tab, human_count)}
  </nav>

  <div class="px-3 py-3 border-t border-[#27272a]">
    <div class="text-xs text-[#52525b] font-medium mb-2">Agents</div>
    <div id="presence-bar" class="space-y-0 max-h-[180px] overflow-y-auto"
         hx-get="/ui/partials/presence" hx-trigger="every 5s" hx-swap="innerHTML">
      {presence_html}
    </div>
  </div>

  <div class="px-3 py-3 border-t border-[#27272a]">
    <div class="flex gap-1.5 items-center">
      <input id="api-key" type="password" placeholder="API key" autocomplete="off"
             class="bg-[#09090b] border border-[#27272a] text-[#fafafa] px-2 py-1.5 rounded-md text-xs w-full placeholder:text-[#52525b] focus:outline-none focus:ring-1 focus:ring-[#3b82f6]/40" />
      <button onclick="saveApiKey()"
              class="bg-[#18181b] border border-[#27272a] text-[#a1a1aa] px-2.5 py-1.5 rounded-md text-xs cursor-pointer shrink-0 hover:text-[#fafafa] hover:bg-[#27272a] transition-colors">Save</button>
    </div>
    <span id="api-key-status" class="text-xs text-[#52525b]"></span>
  </div>
</aside>

<!-- Mobile hamburger -->
<button id="sidebar-toggle"
        class="lg:hidden fixed top-4 left-4 z-50 bg-[#0c0c0e] border border-[#27272a] rounded-lg p-2.5 min-w-[44px] min-h-[44px] text-[#a1a1aa] hover:text-[#fafafa] transition-colors"
        onclick="toggleSidebar()">
  {ICONS["menu"]}
</button>

<!-- Overlay for mobile sidebar -->
<div id="sidebar-overlay" class="fixed inset-0 bg-black/50 z-30 hidden" onclick="toggleSidebar()"></div>

<!-- Loading bar -->
<div id="loading-bar"></div>

<!-- Main content -->
<main class="lg:ml-60 min-h-screen transition-[margin] duration-200">
  <div class="px-4 sm:px-6 py-6 max-w-[1600px]">
    <div id="stats-bar" class="mb-6" hx-get="/ui/partials/stats" hx-trigger="every 5s" hx-swap="innerHTML">
      {stats_html}
    </div>

    <div id="tab-content">
      {content}
    </div>
  </div>
</main>

<script>{CLIENT_JS}</script>
</body>
</html>"""


LANDING_PAGE = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BioAgentics</title>
<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-[#09090b] text-[#fafafa] font-['Inter',system-ui,sans-serif] min-h-screen flex items-center justify-center">
<div class="max-w-2xl mx-auto px-6 py-16 text-center">
  <h1 class="text-4xl font-bold mb-3">BioAgentics</h1>
  <p class="text-[#a1a1aa] text-lg mb-10 max-w-md mx-auto">
    Cancer research coordination platform &mdash; manage research projects, track experiments, and coordinate agent activity.
  </p>
  <div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
    <a href="/ui"
       class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 no-underline text-[#fafafa] hover:border-[#3f3f46] transition-colors shadow-lg min-h-[44px]">
      <div class="text-2xl mb-2 flex justify-center text-[#3b82f6]">{ICONS["chart"]}</div>
      <div class="font-semibold mb-1">Dashboard</div>
      <div class="text-sm text-[#a1a1aa]">View projects, tasks, and agent activity</div>
    </a>
    <a href="/api/docs"
       class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 no-underline text-[#fafafa] hover:border-[#3f3f46] transition-colors shadow-lg min-h-[44px]">
      <div class="text-2xl mb-2 flex justify-center text-[#3b82f6]">{ICONS["book"]}</div>
      <div class="font-semibold mb-1">API Docs</div>
      <div class="text-sm text-[#a1a1aa]">Interactive API documentation</div>
    </a>
    <a href="/ui/projects"
       class="bg-[#0c0c0e] border border-[#27272a] rounded-lg p-6 no-underline text-[#fafafa] hover:border-[#3f3f46] transition-colors shadow-lg min-h-[44px]">
      <div class="text-2xl mb-2 flex justify-center text-[#a1a1aa]">{ICONS["folder"]}</div>
      <div class="font-semibold mb-1">Research</div>
      <div class="text-sm text-[#a1a1aa]">Cancer research projects</div>
    </a>
  </div>
</div>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────


@router.get("/", response_class=HTMLResponse)
def landing_page():
    return LANDING_PAGE


@router.get("/ui")
def ui_redirect():
    return RedirectResponse(url="/ui/dashboard", status_code=302)


@router.get("/ui/dashboard", response_class=HTMLResponse)
def ui_dashboard_page(
    request: Request,
    division: str = Query(default=""),
    db: Session = Depends(get_db),
):
    content = render_dashboard_tab(db, division)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("dashboard", db))
    stats_html = render_stats_html(db, division)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("dashboard", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/projects", response_class=HTMLResponse)
def ui_projects_page(
    request: Request,
    search: str = Query(default="", max_length=200),
    status: str = Query(default=""),
    labels: str = Query(default=""),
    division: str = Query(default=""),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    content = render_projects_tab(db, search, status, labels, offset, division)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("projects", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("projects", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/projects/{name:path}/download")
def ui_project_download(name: str, db: Session = Depends(get_db)):
    """Download a project's output directory as a ZIP file."""
    _repo_resolved = REPO_ROOT.resolve()
    # Look up division from project record
    row = db.execute(select(projects_table).where(projects_table.c.name == name)).first()
    proj_division = row._mapping.get("division", "cancer") if row else "cancer"
    output_dir = (REPO_ROOT / "output" / proj_division / name).resolve()
    if not output_dir.is_dir():
        output_dir = (REPO_ROOT / "output" / name).resolve()
    if not output_dir.is_relative_to(_repo_resolved) or not output_dir.is_dir():
        return HTMLResponse("<h1>Not found</h1>", status_code=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(output_dir.rglob("*")):
            if p.is_file() and p.name != ".DS_Store":
                zf.write(p, arcname=str(p.relative_to(output_dir)))
    buf.seek(0)

    if buf.getbuffer().nbytes <= 22:  # empty zip
        return HTMLResponse("<h1>No files to download</h1>", status_code=404)

    filename = f"{name}-output.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/ui/projects/{name:path}", response_class=HTMLResponse)
def ui_project_detail_page(
    request: Request,
    name: str,
    db: Session = Depends(get_db),
):
    row = db.execute(select(projects_table).where(projects_table.c.name == name)).first()
    if not row:
        not_found = f'<div class="mb-4"><a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors" hx-get="/ui/projects" hx-target="#tab-content" hx-push-url="true">&larr; Back to projects</a></div><div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">Project &ldquo;{esc(name)}&rdquo; not found</div></div>'
        if is_htmx(request):
            return HTMLResponse(not_found + render_sidebar_nav_oob("projects", db))
        stats_html = render_stats_html(db)
        presence_html = render_presence_html(db)
        return HTMLResponse(render_shell("projects", not_found, stats_html, presence_html, _human_task_count(db)))

    # Gather task counts
    tc_rows = db.execute(
        select(tasks.c.status, func.count())
        .where(tasks.c.project == name)
        .group_by(tasks.c.status)
    ).all()
    task_counts = {st: cnt for st, cnt in tc_rows}

    # Journal count
    journal_count = db.execute(
        select(func.count()).select_from(journal).where(journal.c.project == name)
    ).scalar() or 0

    # Recent tasks (10 most recent)
    recent_tasks = db.execute(
        select(tasks)
        .where(tasks.c.project == name)
        .order_by(tasks.c.updated_at.desc())
        .limit(10)
    ).fetchall()

    # Recent journal entries (10 most recent)
    recent_journal = db.execute(
        select(journal)
        .where(journal.c.project == name)
        .order_by(journal.c.created_at.desc())
        .limit(10)
    ).fetchall()

    # Read research artifacts from database, fall back to filesystem
    m = row._mapping
    plan_content = m.get("plan_content") or None
    findings_content = m.get("findings_content") or None

    if not plan_content:
        division = m.get("division") or "cancer"
        # Try division-scoped plans directory first, then legacy PLAN-{name}.md
        plan_path = (REPO_ROOT / "plans" / division / f"{name}.md").resolve()
        if not (plan_path.is_relative_to(REPO_ROOT.resolve()) and plan_path.is_file()):
            plan_path = (REPO_ROOT / f"PLAN-{name}.md").resolve()
        if plan_path.is_relative_to(REPO_ROOT.resolve()) and plan_path.is_file():
            try:
                plan_content = plan_path.read_text()
            except OSError:
                pass

    if not findings_content:
        _repo_resolved = REPO_ROOT.resolve()
        for findings_dir in [REPO_ROOT / "docs" / "findings" / name, REPO_ROOT / "docs" / "findings"]:
            if not findings_dir.resolve().is_relative_to(_repo_resolved):
                continue
            if not findings_dir.is_dir():
                continue
            for md_file in sorted(findings_dir.glob("*.md")):
                if name in md_file.stem:
                    try:
                        findings_content = md_file.read_text()
                    except OSError:
                        pass
                    break
            if findings_content:
                break

    result_files = None
    _repo_resolved = REPO_ROOT.resolve()
    results_dir = (REPO_ROOT / "data" / "results" / name).resolve()
    if results_dir.is_relative_to(_repo_resolved) and results_dir.is_dir():
        result_files = sorted(
            str(p.relative_to(results_dir)) for p in results_dir.rglob("*") if p.is_file()
        )

    # Check for output directory (division-scoped, fallback to legacy)
    proj_division = m.get("division") or "cancer"
    output_dir = (REPO_ROOT / "output" / proj_division / name).resolve()
    if not output_dir.is_dir():
        output_dir = (REPO_ROOT / "output" / name).resolve()
    has_output = output_dir.is_relative_to(_repo_resolved) and output_dir.is_dir() and any(output_dir.rglob("*"))
    output_files = None
    if has_output:
        output_files = sorted(
            str(p.relative_to(output_dir)) for p in output_dir.rglob("*") if p.is_file() and p.name != ".DS_Store"
        )
        has_output = bool(output_files)

    content = render_project_detail(
        row, task_counts, journal_count, recent_tasks, recent_journal,
        plan_content=plan_content, findings_content=findings_content,
        result_files=result_files, output_files=output_files,
    )
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("projects", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("projects", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/tasks", response_class=HTMLResponse)
def ui_tasks_page(
    request: Request,
    search: str = Query(default="", max_length=200),
    username: str = Query(default=""),
    project: str = Query(default=""),
    status: str = Query(default=""),
    priority: str = Query(default=""),
    division: str = Query(default=""),
    sort: str = Query(default="desc"),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    content = render_tasks_tab(db, search, username, project, status, priority, sort, offset, division)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("tasks", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("tasks", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/tasks/{task_id}", response_class=HTMLResponse)
def ui_task_detail_page(
    request: Request,
    task_id: int,
    db: Session = Depends(get_db),
):
    row = db.execute(select(tasks).where(tasks.c.id == task_id)).first()
    if not row:
        not_found = f'<div class="mb-4"><a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors" hx-get="/ui/tasks" hx-target="#tab-content" hx-push-url="true">&larr; Back to tasks</a></div><div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">Task #{task_id} not found</div></div>'
        if is_htmx(request):
            return HTMLResponse(not_found + render_sidebar_nav_oob("tasks", db))
        stats_html = render_stats_html(db)
        presence_html = render_presence_html(db)
        return HTMLResponse(render_shell("tasks", not_found, stats_html, presence_html, _human_task_count(db)))
    content = render_task_detail(row)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("tasks", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("tasks", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/journal", response_class=HTMLResponse)
def ui_journal_page(
    request: Request,
    search: str = Query(default="", max_length=200),
    username: str = Query(default=""),
    project: str = Query(default=""),
    division: str = Query(default=""),
    sort: str = Query(default="desc"),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    content = render_journal_tab(db, search, username, project, sort, offset, division)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("journal", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("journal", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/journal/{entry_id}", response_class=HTMLResponse)
def ui_journal_detail_page(
    request: Request,
    entry_id: int,
    db: Session = Depends(get_db),
):
    row = db.execute(select(journal).where(journal.c.id == entry_id)).first()
    if not row:
        not_found = f'<div class="mb-4"><a class="text-xs text-[#71717a] hover:text-[#3b82f6] cursor-pointer no-underline transition-colors" hx-get="/ui/journal" hx-target="#tab-content" hx-push-url="true">&larr; Back to journal</a></div><div class="flex flex-col items-center justify-center py-16 text-[#a1a1aa]">{ICONS["inbox"]}<div class="mt-3 text-sm font-medium">Journal entry #{entry_id} not found</div></div>'
        if is_htmx(request):
            return HTMLResponse(not_found + render_sidebar_nav_oob("journal", db))
        stats_html = render_stats_html(db)
        presence_html = render_presence_html(db)
        return HTMLResponse(render_shell("journal", not_found, stats_html, presence_html, _human_task_count(db)))
    content = render_journal_detail(row)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("journal", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("journal", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/runs", response_class=HTMLResponse)
def ui_runs_page(
    request: Request,
    agent: str = Query(default=""),
    project: str = Query(default=""),
    division: str = Query(default=""),
    sort: str = Query(default="desc"),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    content = render_runs_tab(db, agent, project, sort, offset, division)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("runs", db))
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    return HTMLResponse(render_shell("runs", content, stats_html, presence_html, _human_task_count(db)))


@router.get("/ui/keys", response_class=HTMLResponse)
def ui_keys_page(
    request: Request,
    x_api_key: str = Header(default=""),
    db: Session = Depends(get_db),
):
    content = render_keys_tab(db, x_api_key)
    if is_htmx(request):
        return HTMLResponse(content + render_sidebar_nav_oob("keys", db))
    # For direct navigation, wrap in shell; keys will re-fetch via htmx if key is in localStorage
    stats_html = render_stats_html(db)
    presence_html = render_presence_html(db)
    shell = render_shell("keys", content, stats_html, presence_html, _human_task_count(db))
    # Inject a script to re-fetch keys with the stored API key
    reload_script = """<script>
if (localStorage.getItem('agent_comms_api_key')) {
  document.addEventListener('DOMContentLoaded', function() {
    htmx.ajax('GET', '/ui/keys', {target: '#tab-content', swap: 'innerHTML'});
  });
}
</script>"""
    shell = shell.replace("</body>", reload_script + "</body>")
    return HTMLResponse(shell)


# ── Partial Endpoints ─────────────────────────────────────────────────


@router.get("/ui/partials/stats", response_class=HTMLResponse)
def ui_partials_stats(division: str = Query(default=""), db: Session = Depends(get_db)):
    return HTMLResponse(render_stats_html(db, division))


@router.get("/ui/partials/presence", response_class=HTMLResponse)
def ui_partials_presence(db: Session = Depends(get_db)):
    return HTMLResponse(render_presence_html(db))


# ── JSON Endpoints (backward compat) ─────────────────────────────────


@router.get("/ui/stats")
def ui_stats(db: Session = Depends(get_db)):
    task_rows = db.execute(
        select(tasks.c.status, func.count()).group_by(tasks.c.status)
    ).all()
    task_counts = {row[0]: row[1] for row in task_rows}
    total_tasks = sum(task_counts.values())
    agent_count = db.execute(select(func.count(agents.c.username.distinct()))).scalar()
    running_agents = db.execute(
        select(func.count(agents.c.username.distinct())).where(agents.c.status == "running")
    ).scalar()
    journal_count = db.execute(select(func.count()).select_from(journal)).scalar()
    human_actionable = db.execute(
        select(func.count())
        .select_from(tasks)
        .where(tasks.c.username == "human")
        .where(tasks.c.status.in_(["pending", "in_progress", "blocked"]))
    ).scalar()
    return {
        "tasks": {
            "total": total_tasks,
            "pending": task_counts.get("pending", 0),
            "in_progress": task_counts.get("in_progress", 0),
            "blocked": task_counts.get("blocked", 0),
            "done": task_counts.get("done", 0),
            "cancelled": task_counts.get("cancelled", 0),
        },
        "human_actionable": human_actionable,
        "agents": {"total": agent_count, "running": running_agents},
        "journal": {"total": journal_count},
    }


@router.get("/ui/filters")
def ui_filters(db: Session = Depends(get_db)):
    usernames = sorted(
        db.execute(
            union(
                select(agents.c.username),
                select(journal.c.username),
                select(tasks.c.username),
            )
        ).scalars().all()
    )
    projects = sorted(
        db.execute(
            union(
                select(agents.c.project).where(agents.c.project.isnot(None)),
                select(journal.c.project).where(journal.c.project.isnot(None)),
                select(tasks.c.project).where(tasks.c.project.isnot(None)),
            )
        ).scalars().all()
    )
    return {"usernames": usernames, "projects": projects}
