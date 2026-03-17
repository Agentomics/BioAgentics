from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioagentics.agent_api.database import SessionLocal, agents, tasks

router = APIRouter(prefix="/status", tags=["status"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("")
def get_status(db: Session = Depends(get_db)):
    # Agents — composite PK is (username, division, project), so pick the most
    # relevant entry per username: prefer "running" over "idle".
    agent_rows = db.execute(select(agents)).fetchall()
    agents_map: dict[str, dict] = {}
    for row in agent_rows:
        m = row._mapping
        username = m["username"]
        division = m["division"] or None
        project = m["project"] or None
        entry: dict = {
            "status": m["status"],
            "last_heartbeat": m["updated_at"],
        }
        if division:
            entry["division"] = division
        if project:
            entry["project"] = project

        if username not in agents_map:
            agents_map[username] = entry
        elif m["status"] == "running" and agents_map[username]["status"] != "running":
            agents_map[username] = entry

    # Task counts by status
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    pending = db.execute(
        select(func.count()).select_from(tasks).where(tasks.c.status == "pending")
    ).scalar()
    in_progress = db.execute(
        select(func.count()).select_from(tasks).where(tasks.c.status == "in_progress")
    ).scalar()
    blocked = db.execute(
        select(func.count()).select_from(tasks).where(tasks.c.status == "blocked")
    ).scalar()
    done_today = db.execute(
        select(func.count())
        .select_from(tasks)
        .where(tasks.c.status == "done")
        .where(tasks.c.updated_at.like(f"{today}%"))
    ).scalar()

    # Blocked tasks detail
    blocked_rows = db.execute(
        select(tasks).where(tasks.c.status == "blocked")
    ).fetchall()
    blocked_tasks = []
    for row in blocked_rows:
        m = row._mapping
        blocked_tasks.append({
            "id": m["id"],
            "title": m["title"],
            "blocked_since": m["blocked_at"],
            "username": m["username"],
        })

    return {
        "agents": agents_map,
        "tasks": {
            "pending": pending,
            "in_progress": in_progress,
            "blocked": blocked,
            "done_today": done_today,
        },
        "blocked_tasks": blocked_tasks,
    }
