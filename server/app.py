from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Observation
from .environment import AegisEnvironment
from pydantic import BaseModel
from typing import Optional

class Action(BaseModel):
    """
    Permissive Catch-all Action wrapper to satisfy openenv-core's ModelValidate expectations.
    This acts merely as a transport layer. The real validation happens inside AegisEnvironment.
    """
    action_type: str = "unknown"
    target_server_id: str = "srv-000"
    
    # Optional fields across all mechanics
    new_tier: Optional[str] = "small"
    confirm_deletion: Optional[bool] = False
    reasoning_trace: Optional[str] = "No reason provided"
    severity_filter: Optional[str] = "info"
    max_entries: Optional[int] = 50

app = create_app(
    AegisEnvironment,
    Action,
    Observation,
    env_name="aegis_env",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
