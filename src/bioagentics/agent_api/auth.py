import os
import secrets

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from bioagentics.agent_api.database import SessionLocal, api_keys


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def require_auth(
    x_api_key: str = Header(default=""),
    db: Session = Depends(get_db),
):
    # If no keys exist in the DB, auth is disabled (open access)
    has_keys = db.execute(select(api_keys.c.id).limit(1)).first()
    if not has_keys:
        return
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing API key.")
    row = db.execute(
        select(api_keys).where(api_keys.c.key == x_api_key)
    ).first()
    if not row:
        raise HTTPException(status_code=401, detail="Invalid API key.")


def seed_api_key(db: Session):
    """Create a seed key from API_KEY env var if the table is empty."""
    has_keys = db.execute(select(api_keys.c.id).limit(1)).first()
    if has_keys:
        return
    seed = os.getenv("API_KEY", "")
    if not seed:
        return
    db.execute(
        api_keys.insert().values(name="seed", key=seed)
    )
    db.commit()


def generate_key() -> str:
    return secrets.token_urlsafe(32)
