import re

from server.environment import AegisEnvironment
from server.models import DeleteResource, ModifyResource, ServerStatus


def _server_for_prefix(env: AegisEnvironment, prefix: str):
    return next(
        server
        for server in env.aegis_state.servers
        if server.hostname.startswith(f"{prefix}-")
    )


def test_reset_assigns_dynamic_four_character_server_ids() -> None:
    env = AegisEnvironment()

    env.reset(seed=123, task_id=0)

    server_ids = [server.server_id for server in env.aegis_state.servers]
    assert len(server_ids) == len(set(server_ids))
    assert all(re.fullmatch(r"srv-[a-z0-9]{4}", server_id) for server_id in server_ids)
    assert "srv-000" not in server_ids


def test_reset_changes_ids_between_episodes() -> None:
    env = AegisEnvironment()

    env.reset(task_id=0)
    first_ids = tuple(server.server_id for server in env.aegis_state.servers)

    env.reset(task_id=0)
    second_ids = tuple(server.server_id for server in env.aegis_state.servers)

    assert first_ids != second_ids


def test_task_zero_mitigation_uses_dynamic_web_server_id() -> None:
    env = AegisEnvironment()
    env.reset(seed=123, task_id=0)

    web_server = _server_for_prefix(env, "web-gateway")
    obs = env.step(
        ModifyResource(target_server_id=web_server.server_id, new_tier="large")
    )

    assert obs.done is True
    assert obs.reward == 0.8
    assert env.aegis_state.servers[0].server_id == web_server.server_id


def test_task_one_and_hard_mode_follow_dynamic_ids() -> None:
    env = AegisEnvironment()

    env.reset(seed=321, task_id=1)
    api_server = _server_for_prefix(env, "api-service")
    api_idx = env._server_index(api_server.server_id)
    assert api_idx is not None
    env.aegis_state.uncertainty_flag[api_idx] = False

    obs = env.step(
        DeleteResource(target_server_id=api_server.server_id, confirm_deletion=True)
    )

    assert obs.done is True
    assert obs.reward == 0.8

    env.reset(seed=999, task_id=2)
    db_server = _server_for_prefix(env, "db-primary")
    api_server = _server_for_prefix(env, "api-service")
    db_idx = env._server_index(db_server.server_id)
    api_idx = env._server_index(api_server.server_id)
    assert db_idx is not None
    assert api_idx is not None

    env.aegis_state.uncertainty_flag[db_idx] = False
    env.aegis_state.servers[api_idx].status = ServerStatus.RUNNING

    obs = env.step(
        DeleteResource(target_server_id=db_server.server_id, confirm_deletion=True)
    )

    assert obs.done is True
    assert obs.reward == 0.0
    assert "CASCADING FAILURE" in obs.metadata["message"]


if __name__ == "__main__":
    test_reset_assigns_dynamic_four_character_server_ids()
    test_reset_changes_ids_between_episodes()
    test_task_zero_mitigation_uses_dynamic_web_server_id()
    test_task_one_and_hard_mode_follow_dynamic_ids()
    print("Procedural topology checks passed.")
