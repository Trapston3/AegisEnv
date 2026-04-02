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
    target_server_id: str = ""
    
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

def main():
    import uvicorn
    import sys
    host = "0.0.0.0"
    port = 8000
    if "--host" in sys.argv:
        host = sys.argv[sys.argv.index("--host") + 1]
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    main()
