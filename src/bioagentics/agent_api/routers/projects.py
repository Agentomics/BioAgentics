from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioagentics.agent_api.auth import require_auth
from bioagentics.agent_api.database import SessionLocal, projects_table
from bioagentics.agent_api.models import ProjectCreate, ProjectEntry, ProjectList, ProjectSummary, ProjectUpdate

router = APIRouter(prefix="/projects", tags=["projects"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", status_code=201, response_model=ProjectEntry, dependencies=[Depends(require_auth)])
def create_project(body: ProjectCreate, db: Session = Depends(get_db)):
    existing = db.execute(
        select(projects_table).where(projects_table.c.name == body.name)
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Project already exists.")

    db.execute(
        projects_table.insert().values(
            name=body.name,
            status=body.status,
            description=body.description,
            labels=body.labels,
            plan_content=body.plan_content,
            findings_content=body.findings_content,
        )
    )
    db.commit()
    row = db.execute(
        select(projects_table).where(projects_table.c.name == body.name)
    ).first()
    return row._mapping


@router.get("", response_model=ProjectList)
def list_projects(
    status: str | None = Query(default=None),
    labels: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    summary_cols = [
        projects_table.c.name,
        projects_table.c.status,
        projects_table.c.description,
        projects_table.c.labels,
        projects_table.c.created_at,
        projects_table.c.updated_at,
    ]
    query = select(*summary_cols)
    count_query = select(func.count()).select_from(projects_table)

    if status is not None:
        query = query.where(projects_table.c.status == status)
        count_query = count_query.where(projects_table.c.status == status)
    if labels is not None:
        query = query.where(projects_table.c.labels.contains(labels))
        count_query = count_query.where(projects_table.c.labels.contains(labels))

    total = db.execute(count_query).scalar()
    rows = db.execute(
        query.order_by(projects_table.c.created_at.desc()).limit(limit).offset(offset)
    ).fetchall()

    return ProjectList(
        total=total,
        items=[row._mapping for row in rows],
    )


@router.get("/{name}", response_model=ProjectEntry)
def get_project(name: str, db: Session = Depends(get_db)):
    row = db.execute(
        select(projects_table).where(projects_table.c.name == name)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    return row._mapping


@router.patch("/{name}", response_model=ProjectEntry, dependencies=[Depends(require_auth)])
def update_project(name: str, body: ProjectUpdate, db: Session = Depends(get_db)):
    row = db.execute(
        select(projects_table).where(projects_table.c.name == name)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found.")

    updates = body.model_dump(exclude_unset=True)
    if not updates:
        return row._mapping

    updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.execute(
        projects_table.update().where(projects_table.c.name == name).values(**updates)
    )
    db.commit()

    row = db.execute(
        select(projects_table).where(projects_table.c.name == name)
    ).first()
    return row._mapping


@router.delete("/{name}", status_code=204, dependencies=[Depends(require_auth)])
def delete_project(name: str, db: Session = Depends(get_db)):
    row = db.execute(
        select(projects_table).where(projects_table.c.name == name)
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found.")
    db.execute(projects_table.delete().where(projects_table.c.name == name))
    db.commit()
