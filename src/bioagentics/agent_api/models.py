from pydantic import BaseModel, Field


class JournalCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, pattern=r"^\S+$")
    project: str | None = Field(default=None, min_length=1, max_length=64)
    content: str = Field(..., min_length=1, max_length=10_000)


class JournalEntry(BaseModel):
    id: int
    username: str
    project: str | None
    content: str
    created_at: str


class JournalList(BaseModel):
    total: int
    items: list[JournalEntry]


VALID_AGENT_STATUSES = ("running", "idle")
_AGENT_STATUS_PATTERN = r"^(running|idle)$"


class AgentRegister(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, pattern=r"^\S+$")
    status: str = Field(default="running", pattern=_AGENT_STATUS_PATTERN)
    project: str | None = Field(default=None, min_length=1, max_length=64)


class AgentEntry(BaseModel):
    username: str
    status: str
    project: str | None
    started_at: str
    updated_at: str


class AgentList(BaseModel):
    total: int
    items: list[AgentEntry]


VALID_STATUSES = ("pending", "in_progress", "blocked", "done", "cancelled")
_STATUS_PATTERN = r"^(pending|in_progress|blocked|done|cancelled)$"


class TaskCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=64, pattern=r"^\S+$")
    project: str | None = Field(default=None, min_length=1, max_length=64)
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=5_000)
    status: str = Field(default="pending", pattern=_STATUS_PATTERN)
    priority: int = Field(default=1, ge=1, le=5)
    blocked_reason: str | None = Field(default=None, max_length=1000)


class TaskUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    status: str | None = Field(default=None, pattern=_STATUS_PATTERN)
    description: str | None = Field(default=None, max_length=5_000)
    priority: int | None = Field(default=None, ge=1, le=5)
    blocked_reason: str | None = Field(default=None, max_length=1000)


class TaskEntry(BaseModel):
    id: int
    username: str
    project: str | None
    title: str
    description: str | None
    status: str
    priority: int
    created_at: str
    updated_at: str
    blocked_at: str | None
    blocked_reason: str | None


class TaskList(BaseModel):
    total: int
    items: list[TaskEntry]


class ApiKeyCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)


class ApiKeyEntry(BaseModel):
    id: int
    name: str
    key: str
    created_at: str


class ApiKeyList(BaseModel):
    total: int
    items: list[ApiKeyEntry]


class RunCreate(BaseModel):
    agent: str = Field(..., min_length=1, max_length=64)
    backend: str = Field(..., min_length=1, max_length=64)
    model: str = Field(..., min_length=1, max_length=128)
    project: str | None = Field(default=None, min_length=1, max_length=64)
    started_at: str = Field(..., min_length=1, max_length=64)
    finished_at: str | None = Field(default=None, max_length=64)
    exit_code: int | None = None
    duration_seconds: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class RunUpdate(BaseModel):
    finished_at: str | None = Field(default=None, max_length=64)
    exit_code: int | None = None
    tasks_completed: int | None = Field(default=None, ge=0)
    duration_seconds: int | None = Field(default=None, ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_creation_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)


class RunEntry(BaseModel):
    id: int
    agent: str
    backend: str
    model: str
    project: str | None
    started_at: str
    finished_at: str | None
    exit_code: int | None
    tasks_completed: int
    duration_seconds: int | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_creation_tokens: int | None
    cost_usd: float | None


class RunList(BaseModel):
    total: int
    items: list[RunEntry]


VALID_PROJECT_STATUSES = (
    "proposed", "planning", "development", "analysis",
    "validation", "documentation", "published", "cancelled",
)
_PROJECT_STATUS_PATTERN = (
    r"^(proposed|planning|development|analysis|validation|documentation|published|cancelled)$"
)


class ProjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, pattern=r"^\S+$")
    status: str = Field(default="proposed", pattern=_PROJECT_STATUS_PATTERN)
    description: str | None = Field(default=None, max_length=5_000)
    labels: str | None = Field(default=None, max_length=500)


class ProjectUpdate(BaseModel):
    status: str | None = Field(default=None, pattern=_PROJECT_STATUS_PATTERN)
    description: str | None = Field(default=None, max_length=5_000)
    labels: str | None = Field(default=None, max_length=500)


class ProjectEntry(BaseModel):
    name: str
    status: str
    description: str | None
    labels: str | None
    created_at: str
    updated_at: str


class ProjectList(BaseModel):
    total: int
    items: list[ProjectEntry]
