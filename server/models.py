"""
AegisEnv — Pydantic models for the OpenEnv-core RL environment.

Defines strictly typed State, Observation, and Action schemas for a
simulated cloud infrastructure management agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
#  Enums
# ──────────────────────────────────────────────────────────────────────

class ServerStatus(str, Enum):
    """Operational status of a cloud server."""
    RUNNING = "running"
    STOPPED = "stopped"
    DEGRADED = "degraded"
    PROVISIONING = "provisioning"
    TERMINATED = "terminated"


class ResourceTier(str, Enum):
    """Available compute tiers for a server resource."""
    SMALL = "small"      # 2 vCPU / 4 GB
    MEDIUM = "medium"    # 4 vCPU / 16 GB
    LARGE = "large"      # 8 vCPU / 32 GB
    XLARGE = "xlarge"    # 16 vCPU / 64 GB


class LogSeverity(str, Enum):
    """Log severity levels that can be queried."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ──────────────────────────────────────────────────────────────────────
#  State  (the full ground-truth of the environment)
# ──────────────────────────────────────────────────────────────────────

class ServerRecord(BaseModel):
    """A single server entry in the cloud infrastructure database."""

    server_id: str = Field(
        ...,
        description="Unique identifier for the server (e.g. 'srv-001' or dynamically generated 'srv-web-7b2').",
        min_length=1,
    )
    hostname: str = Field(
        ...,
        description="DNS hostname of the server.",
        min_length=1,
        max_length=253,
    )
    status: ServerStatus = Field(
        default=ServerStatus.RUNNING,
        description="Current operational status.",
    )
    tier: ResourceTier = Field(
        default=ResourceTier.SMALL,
        description="Compute tier allocated to this server.",
    )
    cpu_utilisation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Current CPU utilisation percentage (0-100).",
    )
    memory_utilisation: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Current memory utilisation percentage (0-100).",
    )
    open_incident_count: int = Field(
        default=0,
        ge=0,
        description="Number of unresolved incidents on this server.",
    )

    class Config:
        use_enum_values = True


class AegisState(BaseModel):
    """
    Full ground-truth state of the AegisEnv cloud infrastructure.

    Represents a fake cloud database with exactly 5 server records and
    an uncertainty_flag array that the agent must reason about.
    """

    episode_id: str = Field(
        default="",
        description="Unique identifier for the current episode.",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="Number of steps taken in the current episode.",
    )

    # ── Cloud infrastructure database ────────────────────────────────
    servers: list[ServerRecord] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 server records making up the cloud infra DB.",
    )

    # ── Uncertainty flags ────────────────────────────────────────────
    uncertainty_flag: list[bool] = Field(
        ...,
        min_length=5,
        max_length=5,
        description=(
            "Per-server boolean flags. True indicates that the "
            "corresponding server's telemetry data may be unreliable "
            "and the agent should consider requesting human confirmation "
            "before taking destructive actions."
        ),
    )

    # ── Global metadata ──────────────────────────────────────────────
    global_alert_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Infrastructure-wide alert level (0=nominal … 4=critical).",
    )
    budget_remaining: float = Field(
        default=1000.0,
        ge=0.0,
        description="Remaining operational budget for the episode (USD).",
    )

    class Config:
        use_enum_values = True


# ──────────────────────────────────────────────────────────────────────
#  Observation  (partial / noisy view the agent receives)
# ──────────────────────────────────────────────────────────────────────

class ServerSnapshot(BaseModel):
    """Observable slice of a single server — may be stale or partial."""

    server_id: str = Field(..., min_length=1)
    hostname: str
    status: ServerStatus
    tier: ResourceTier
    cpu_utilisation: float = Field(ge=0.0, le=100.0)
    memory_utilisation: float = Field(ge=0.0, le=100.0)
    is_uncertain: bool = Field(
        default=False,
        description="Mirrors the uncertainty_flag for this server.",
    )

    class Config:
        use_enum_values = True


class AegisObservation(BaseModel):
    """
    What the agent observes after each step.

    This is a *partial* projection of the full AegisState: the agent
    sees snapshots of the 5 servers, cumulative reward so far, and a
    human-readable message describing the last action's outcome.
    """

    server_snapshots: list[ServerSnapshot] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Observable snapshots of all 5 servers.",
    )
    message: str = Field(
        default="",
        description="Human-readable feedback from the environment.",
    )
    cumulative_reward: float = Field(
        default=0.0,
        description="Total reward accumulated so far in this episode.",
    )
    global_alert_level: int = Field(
        default=0,
        ge=0,
        le=4,
        description="Current infrastructure-wide alert level.",
    )

    class Config:
        use_enum_values = True

# ──────────────────────────────────────────────────────────────────────
#  Actions  (Robust defaults to prevent 422 errors)
# ──────────────────────────────────────────────────────────────────────

class ModifyResource(BaseModel):
    """Scale a server up/down by changing its compute tier."""
    action_type: Literal["modify_resource"] = "modify_resource"
    target_server_id: str = Field(..., min_length=1)
    # Added default to prevent 422
    new_tier: ResourceTier = Field(default=ResourceTier.SMALL)

    class Config:
        use_enum_values = True


class DeleteResource(BaseModel):
    """Permanently terminate a server instance."""
    action_type: Literal["delete_resource"] = "delete_resource"
    target_server_id: str = Field(..., min_length=1)
    # Added default to prevent 422
    confirm_deletion: bool = Field(default=False)


class QueryLogs(BaseModel):
    """Retrieve recent log entries for a specific server."""
    action_type: Literal["query_logs"] = "query_logs"
    target_server_id: str = Field(..., min_length=1)
    severity_filter: LogSeverity = Field(default=LogSeverity.INFO)
    max_entries: int = Field(default=50, ge=1, le=500)

    class Config:
        use_enum_values = True


class RequestHumanConfirmation(BaseModel):
    """Pause and escalate to a human operator."""
    action_type: Literal["request_human_confirmation"] = "request_human_confirmation"
    # Reduced min_length to 1 to be more forgiving to short LLM responses
    reasoning_trace: str = Field(default="No reason provided", min_length=1)

class UnknownAction(BaseModel):
    action_type: Literal["unknown"] = "unknown"
    target_server_id: str = "srv-000"

# ── Discriminated union of all actions ───────────────────────────────

AegisAction = Annotated[
    Union[
        ModifyResource,
        DeleteResource,
        QueryLogs,
        RequestHumanConfirmation,
        UnknownAction,
    ],
    Field(discriminator="action_type"),
]
"""
Top-level action type consumed by `AegisEnvironment.step()`.

Uses a discriminated union keyed on `action_type` so Pydantic can
parse raw JSON into the correct concrete action with zero ambiguity.

Example
-------
>>> from pydantic import TypeAdapter
>>> ta = TypeAdapter(AegisAction)
>>> ta.validate_python({"action_type": "query_logs", "target_server_id": "srv-002"})
QueryLogs(action_type='query_logs', target_server_id='srv-002', ...)
"""


# ──────────────────────────────────────────────────────────────────────
#  Convenience: default 5-server state factory
# ──────────────────────────────────────────────────────────────────────

_DEFAULT_HOSTNAMES = [
    "web-gateway-alpha",
    "api-service-beta",
    "db-primary-gamma",
    "cache-layer-delta",
    "worker-pool-epsilon",
]


def make_default_state(episode_id: str = "") -> AegisState:
    """Return a fresh AegisState with 5 idle servers and no uncertainty."""
    servers = [
        ServerRecord(
            server_id=f"srv-{i:03d}",
            hostname=_DEFAULT_HOSTNAMES[i],
            status=ServerStatus.RUNNING,
            tier=ResourceTier.SMALL,
            cpu_utilisation=0.0,
            memory_utilisation=0.0,
            open_incident_count=0,
        )
        for i in range(5)
    ]
    return AegisState(
        episode_id=episode_id,
        step_count=0,
        servers=servers,
        uncertainty_flag=[False] * 5,
        global_alert_level=0,
        budget_remaining=1000.0,
    )
