import os

from sqlalchemy import (
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./db.sqlite")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)

metadata = MetaData()

journal = Table(
    "journal",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, nullable=False),
    Column("division", String, nullable=True),
    Column("project", String, nullable=True),
    Column("content", Text, nullable=False),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
)

tasks = Table(
    "tasks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", String, nullable=False),
    Column("division", String, nullable=True),
    Column("project", String, nullable=True),
    Column("title", String, nullable=False),
    Column("description", Text, nullable=True),
    Column("status", String, nullable=False, server_default=text("'pending'")),
    Column("priority", Integer, nullable=False, server_default=text("1")),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
    Column(
        "updated_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
    Column("blocked_at", String, nullable=True),
    Column("blocked_reason", String, nullable=True),
)


agents = Table(
    "agents",
    metadata,
    Column("username", String, primary_key=True),
    Column("division", String, primary_key=True, server_default=text("''")),
    Column("project", String, primary_key=True, server_default=text("''")),
    Column("status", String, nullable=False, server_default=text("'running'")),
    Column(
        "started_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
    Column(
        "updated_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
)


api_keys = Table(
    "api_keys",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("key", String, nullable=False, unique=True),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
)


runs = Table(
    "runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent", String, nullable=False),
    Column("backend", String, nullable=False),
    Column("model", String, nullable=False),
    Column("division", String, nullable=True),
    Column("project", String, nullable=True),
    Column("started_at", String, nullable=False),
    Column("finished_at", String, nullable=True),
    Column("exit_code", Integer, nullable=True),
    Column("tasks_completed", Integer, nullable=False, server_default=text("0")),
    Column("duration_seconds", Integer, nullable=True),
    Column("input_tokens", Integer, nullable=True),
    Column("output_tokens", Integer, nullable=True),
    Column("cache_read_tokens", Integer, nullable=True),
    Column("cache_creation_tokens", Integer, nullable=True),
    Column("cost_usd", Float, nullable=True),
)


projects_table = Table(
    "projects",
    metadata,
    Column("name", String, primary_key=True),
    Column("division", String, nullable=True),
    Column("status", String, nullable=False, server_default=text("'proposed'")),
    Column("description", String, nullable=True),
    Column("labels", String, nullable=True),
    Column("plan_content", Text, nullable=True),
    Column("findings_content", Text, nullable=True),
    Column("plain_summary", Text, nullable=True),
    Column("impact_score", String, nullable=True),
    Column(
        "created_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
    Column(
        "updated_at",
        String,
        nullable=False,
        server_default=text("(strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))"),
    ),
)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("PRAGMA journal_mode=WAL"))
        # Migrate agents table: PK is now (username, division, project).
        # Recreate if schema is outdated (presence data is ephemeral).
        try:
            cols = conn.execute(text("PRAGMA table_info(agents)")).fetchall()
            col_names = [c[1] for c in cols]
            if "division" not in col_names:
                conn.execute(text("DROP TABLE IF EXISTS agents"))
                conn.commit()
        except Exception:
            pass
        # Migrate runs table: add cache token columns if missing
        try:
            cols = conn.execute(text("PRAGMA table_info(runs)")).fetchall()
            col_names = [c[1] for c in cols]
            if "cache_read_tokens" not in col_names:
                conn.execute(text("ALTER TABLE runs ADD COLUMN cache_read_tokens INTEGER"))
                conn.execute(text("ALTER TABLE runs ADD COLUMN cache_creation_tokens INTEGER"))
                conn.commit()
        except Exception:
            pass
        # Migrate projects table: add plan_content and findings_content if missing
        try:
            cols = conn.execute(text("PRAGMA table_info(projects)")).fetchall()
            col_names = [c[1] for c in cols]
            if "plan_content" not in col_names:
                conn.execute(text("ALTER TABLE projects ADD COLUMN plan_content TEXT"))
                conn.execute(text("ALTER TABLE projects ADD COLUMN findings_content TEXT"))
                conn.commit()
        except Exception:
            pass
        # Migrate projects table: add plain_summary and impact_score if missing
        try:
            cols = conn.execute(text("PRAGMA table_info(projects)")).fetchall()
            col_names = [c[1] for c in cols]
            if "plain_summary" not in col_names:
                conn.execute(text("ALTER TABLE projects ADD COLUMN plain_summary TEXT"))
                conn.execute(text("ALTER TABLE projects ADD COLUMN impact_score TEXT"))
                conn.commit()
        except Exception:
            pass
        # Migrate all tables: add division column if missing
        for tbl_name in ("journal", "tasks", "runs", "projects"):
            try:
                cols = conn.execute(text(f"PRAGMA table_info({tbl_name})")).fetchall()
                col_names = [c[1] for c in cols]
                if "division" not in col_names:
                    conn.execute(text(f"ALTER TABLE {tbl_name} ADD COLUMN division TEXT"))
                    conn.execute(text(f"UPDATE {tbl_name} SET division = 'cancer' WHERE division IS NULL"))
                    conn.commit()
            except Exception:
                pass
    metadata.create_all(engine)
