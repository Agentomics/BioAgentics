from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioagentics.agent_api.auth import require_auth
from bioagentics.agent_api.database import SessionLocal, agents
from bioagentics.agent_api.models import AgentEntry, AgentList, AgentRegister

router = APIRouter(prefix="/agents", tags=["agents"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", status_code=200, response_model=AgentEntry, dependencies=[Depends(require_auth)])
def register_agent(body: AgentRegister, db: Session = Depends(get_db)):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    division = body.division or ""
    project = body.project or ""
    pk_cond = (
        (agents.c.username == body.username)
        & (agents.c.division == division)
        & (agents.c.project == project)
    )
    existing = db.execute(select(agents).where(pk_cond)).first()

    if existing:
        db.execute(
            agents.update().where(pk_cond).values(status=body.status, updated_at=now)
        )
    else:
        db.execute(
            agents.insert().values(
                username=body.username,
                division=division,
                status=body.status,
                project=project,
                started_at=now,
                updated_at=now,
            )
        )
    db.commit()

    row = db.execute(select(agents).where(pk_cond)).first()
    return row._mapping


@router.get("", response_model=AgentList)
def list_agents(
    status: str | None = Query(default=None),
    division: str | None = Query(default=None),
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = select(agents)
    count_query = select(func.count()).select_from(agents)

    if status is not None:
        query = query.where(agents.c.status == status)
        count_query = count_query.where(agents.c.status == status)
    if division is not None:
        query = query.where(agents.c.division == division)
        count_query = count_query.where(agents.c.division == division)
    if project is not None:
        query = query.where(agents.c.project == project)
        count_query = count_query.where(agents.c.project == project)

    total = db.execute(count_query).scalar()
    rows = db.execute(query.order_by(agents.c.updated_at.desc())).fetchall()

    return AgentList(
        total=total,
        items=[row._mapping for row in rows],
    )


@router.get("/{username}", response_model=AgentEntry)
def get_agent(
    username: str,
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = select(agents).where(agents.c.username == username)
    if project is not None:
        query = query.where(agents.c.project == project)
    row = db.execute(query.order_by(agents.c.updated_at.desc())).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    return row._mapping


@router.delete("/{username}", status_code=204, dependencies=[Depends(require_auth)])
def deregister_agent(
    username: str,
    project: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    cond = agents.c.username == username
    if project is not None:
        cond = cond & (agents.c.project == project)
    row = db.execute(select(agents).where(cond)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Agent not found.")
    db.execute(agents.delete().where(cond))
    db.commit()
