import os
from contextlib import asynccontextmanager

import bioagentics.config  # noqa: F401 — loads .env into os.environ

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from bioagentics.agent_api.auth import seed_api_key
from bioagentics.agent_api.database import SessionLocal, init_db
from bioagentics.agent_api.routers import agents, journal, keys, projects, runs, status, tasks, ui


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), camera=(), microphone=()"
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    db = SessionLocal()
    try:
        seed_api_key(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="Agent Communication API",
    version="1.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "").split(",") if os.getenv("CORS_ORIGINS") else ["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(agents.router, prefix="/api")
app.include_router(journal.router, prefix="/api")
app.include_router(keys.router, prefix="/api")
app.include_router(projects.router, prefix="/api")
app.include_router(runs.router, prefix="/api")
app.include_router(status.router, prefix="/api")
app.include_router(tasks.router, prefix="/api")
app.include_router(ui.router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "bioagentics.agent_api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info"),
        reload=True,
    )
