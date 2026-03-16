from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent_api.auth import require_auth
from agent_api.database import SessionLocal, runs
from agent_api.models import RunCreate, RunEntry, RunList, RunUpdate

router = APIRouter(prefix="/runs", tags=["runs"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", status_code=201, response_model=RunEntry, dependencies=[Depends(require_auth)])
def create_run(body: RunCreate, db: Session = Depends(get_db)):
    values = {
        "agent": body.agent,
        "backend": body.backend,
        "model": body.model,
        "project": body.project,
        "started_at": body.started_at,
    }
    # Accept optional completion fields so runs can be logged in a single POST
    if body.finished_at is not None:
        values["finished_at"] = body.finished_at
    if body.exit_code is not None:
        values["exit_code"] = body.exit_code
    if body.duration_seconds is not None:
        values["duration_seconds"] = body.duration_seconds
    if body.input_tokens is not None:
        values["input_tokens"] = body.input_tokens
    if body.output_tokens is not None:
        values["output_tokens"] = body.output_tokens
    if body.cost_usd is not None:
        values["cost_usd"] = body.cost_usd

    result = db.execute(runs.insert().values(**values))
    db.commit()
    row = db.execute(
        select(runs).where(runs.c.id == result.inserted_primary_key[0])
    ).first()
    return row._mapping


@router.get("", response_model=RunList)
def list_runs(
    agent: str | None = Query(default=None),
    project: str | None = Query(default=None),
    sort: Literal["asc", "desc"] = Query(default="desc"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = select(runs)
    count_query = select(func.count()).select_from(runs)

    if agent is not None:
        query = query.where(runs.c.agent == agent)
        count_query = count_query.where(runs.c.agent == agent)
    if project is not None:
        query = query.where(runs.c.project == project)
        count_query = count_query.where(runs.c.project == project)

    total = db.execute(count_query).scalar()
    order = runs.c.started_at.asc() if sort == "asc" else runs.c.started_at.desc()
    rows = db.execute(
        query.order_by(order).limit(limit).offset(offset)
    ).fetchall()

    return RunList(
        total=total,
        items=[row._mapping for row in rows],
    )


@router.get("/{run_id}", response_model=RunEntry)
def get_run(run_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(runs).where(runs.c.id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    return row._mapping


@router.delete("/{run_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_run(run_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(runs).where(runs.c.id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")
    db.execute(runs.delete().where(runs.c.id == run_id))
    db.commit()


@router.patch("/{run_id}", response_model=RunEntry, dependencies=[Depends(require_auth)])
def update_run(run_id: int, body: RunUpdate, db: Session = Depends(get_db)):
    row = db.execute(select(runs).where(runs.c.id == run_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found.")

    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return row._mapping

    db.execute(runs.update().where(runs.c.id == run_id).values(**updates))
    db.commit()

    row = db.execute(select(runs).where(runs.c.id == run_id)).first()
    return row._mapping
