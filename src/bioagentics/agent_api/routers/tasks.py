import re
from datetime import datetime, timedelta, timezone

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioagentics.agent_api.auth import require_auth
from bioagentics.agent_api.database import SessionLocal, tasks
from bioagentics.agent_api.models import TaskCreate, TaskEntry, TaskList, TaskUpdate


def _parse_duration(value: str) -> timedelta:
    """Parse a duration string like '6h', '24h', '1h' into a timedelta."""
    match = re.fullmatch(r"(\d+)h", value)
    if not match:
        raise ValueError(f"Invalid duration format: {value!r}. Use format like '6h', '24h'.")
    return timedelta(hours=int(match.group(1)))

router = APIRouter(prefix="/tasks", tags=["tasks"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", status_code=201, response_model=TaskEntry, dependencies=[Depends(require_auth)])
def create_task(body: TaskCreate, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    blocked_at = now if body.status == "blocked" else None
    result = db.execute(
        tasks.insert().values(
            username=body.username,
            project=body.project,
            title=body.title,
            description=body.description,
            status=body.status,
            priority=body.priority,
            blocked_reason=body.blocked_reason,
            blocked_at=blocked_at,
        )
    )
    db.commit()
    row = db.execute(
        select(tasks).where(tasks.c.id == result.inserted_primary_key[0])
    ).first()
    return row._mapping


@router.get("", response_model=TaskList)
def list_tasks(
    username: str | None = Query(default=None),
    project: str | None = Query(default=None),
    status: str | None = Query(default=None),
    priority: int | None = Query(default=None, ge=1, le=5),
    older_than: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
    sort: Literal["asc", "desc"] = Query(default="desc"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = select(tasks)
    count_query = select(func.count()).select_from(tasks)

    if username is not None:
        query = query.where(tasks.c.username == username)
        count_query = count_query.where(tasks.c.username == username)
    if project is not None:
        query = query.where(tasks.c.project == project)
        count_query = count_query.where(tasks.c.project == project)
    if status is not None:
        query = query.where(tasks.c.status == status)
        count_query = count_query.where(tasks.c.status == status)
    if priority is not None:
        query = query.where(tasks.c.priority == priority)
        count_query = count_query.where(tasks.c.priority == priority)
    if search is not None:
        term = f"%{search}%"
        cond = tasks.c.title.like(term) | tasks.c.description.like(term)
        query = query.where(cond)
        count_query = count_query.where(cond)
    if older_than is not None and status == "blocked":
        try:
            delta = _parse_duration(older_than)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        cutoff = (datetime.now(timezone.utc) - delta).strftime("%Y-%m-%dT%H:%M:%SZ")
        query = query.where(tasks.c.blocked_at < cutoff)
        count_query = count_query.where(tasks.c.blocked_at < cutoff)

    total = db.execute(count_query).scalar()
    date_order = tasks.c.created_at.asc() if sort == "asc" else tasks.c.created_at.desc()
    rows = db.execute(
        query.order_by(tasks.c.priority.desc(), date_order)
        .limit(limit)
        .offset(offset)
    ).fetchall()

    return TaskList(
        total=total,
        items=[row._mapping for row in rows],
    )


@router.get("/{task_id}", response_model=TaskEntry)
def get_task(task_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(tasks).where(tasks.c.id == task_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    return row._mapping


@router.patch("/{task_id}", response_model=TaskEntry, dependencies=[Depends(require_auth)])
def update_task(task_id: int, body: TaskUpdate, db: Session = Depends(get_db)):
    row = db.execute(select(tasks).where(tasks.c.id == task_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found.")

    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return row._mapping

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updates["updated_at"] = now

    if "status" in updates:
        if updates["status"] == "blocked" and row._mapping["status"] != "blocked":
            updates["blocked_at"] = now
        elif updates["status"] != "blocked" and row._mapping["status"] == "blocked":
            updates["blocked_at"] = None

    db.execute(tasks.update().where(tasks.c.id == task_id).values(**updates))
    db.commit()

    row = db.execute(select(tasks).where(tasks.c.id == task_id)).first()
    return row._mapping


@router.delete("/{task_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_task(task_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(tasks).where(tasks.c.id == task_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Task not found.")
    db.execute(tasks.delete().where(tasks.c.id == task_id))
    db.commit()
