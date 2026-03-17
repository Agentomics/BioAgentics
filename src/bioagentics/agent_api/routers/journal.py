from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from bioagentics.agent_api.auth import require_auth
from bioagentics.agent_api.database import SessionLocal, journal
from bioagentics.agent_api.models import JournalCreate, JournalEntry, JournalList

router = APIRouter(prefix="/journal", tags=["journal"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("", status_code=201, response_model=JournalEntry, dependencies=[Depends(require_auth)])
def create_journal_entry(body: JournalCreate, db: Session = Depends(get_db)):
    result = db.execute(
        journal.insert().values(
            username=body.username,
            division=body.division,
            project=body.project,
            content=body.content,
        )
    )
    db.commit()
    row = db.execute(
        select(journal).where(journal.c.id == result.inserted_primary_key[0])
    ).first()
    return row._mapping


@router.get("", response_model=JournalList)
def list_journal_entries(
    username: str | None = Query(default=None),
    division: str | None = Query(default=None),
    project: str | None = Query(default=None),
    search: str | None = Query(default=None, max_length=200),
    sort: Literal["asc", "desc"] = Query(default="desc"),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    query = select(journal)
    count_query = select(func.count()).select_from(journal)

    if username is not None:
        query = query.where(journal.c.username == username)
        count_query = count_query.where(journal.c.username == username)
    if division is not None:
        query = query.where(journal.c.division == division)
        count_query = count_query.where(journal.c.division == division)
    if project is not None:
        query = query.where(journal.c.project == project)
        count_query = count_query.where(journal.c.project == project)
    if search is not None:
        cond = journal.c.content.contains(search)
        query = query.where(cond)
        count_query = count_query.where(cond)

    total = db.execute(count_query).scalar()
    order = journal.c.created_at.asc() if sort == "asc" else journal.c.created_at.desc()
    rows = db.execute(
        query.order_by(order).limit(limit).offset(offset)
    ).fetchall()

    return JournalList(
        total=total,
        items=[row._mapping for row in rows],
    )
