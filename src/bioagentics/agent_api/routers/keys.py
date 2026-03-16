from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from agent_api.auth import generate_key, require_auth
from agent_api.database import SessionLocal, api_keys
from agent_api.models import ApiKeyCreate, ApiKeyEntry, ApiKeyList

router = APIRouter(prefix="/keys", tags=["keys"])


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.get("", response_model=ApiKeyList, dependencies=[Depends(require_auth)])
def list_keys(db: Session = Depends(get_db)):
    total = db.execute(select(func.count()).select_from(api_keys)).scalar()
    rows = db.execute(
        select(api_keys).order_by(api_keys.c.created_at.desc())
    ).fetchall()
    # Mask keys in listing — only show first 8 chars
    items = []
    for row in rows:
        m = row._mapping
        items.append(ApiKeyEntry(
            id=m["id"],
            name=m["name"],
            key=m["key"][:8] + "...",
            created_at=m["created_at"],
        ))
    return ApiKeyList(total=total, items=items)


@router.post("", status_code=201, response_model=ApiKeyEntry, dependencies=[Depends(require_auth)])
def create_key(body: ApiKeyCreate, db: Session = Depends(get_db)):
    key = generate_key()
    result = db.execute(
        api_keys.insert().values(name=body.name, key=key)
    )
    db.commit()
    row = db.execute(
        select(api_keys).where(api_keys.c.id == result.inserted_primary_key[0])
    ).first()
    return row._mapping


@router.delete("/{key_id}", status_code=204, dependencies=[Depends(require_auth)])
def delete_key(key_id: int, db: Session = Depends(get_db)):
    row = db.execute(select(api_keys).where(api_keys.c.id == key_id)).first()
    if row is None:
        raise HTTPException(status_code=404, detail="API key not found.")
    db.execute(api_keys.delete().where(api_keys.c.id == key_id))
    db.commit()
