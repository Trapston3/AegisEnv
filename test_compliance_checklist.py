import asyncio

from server.environment import AegisEnvironment
from server.models import (
    AegisObservation,
    DeleteResource,
    ModifyResource,
    QueryLogs,
)


class AsyncBonusEnvironment(AegisEnvironment):
    async def _evaluate_reasoning_bonus_async(self) -> float:
        return 0.2

    def _evaluate_reasoning_bonus_sync(self) -> float:
        return 0.2


def main() -> None:
    env = AegisEnvironment()

    obs = env.reset(seed=11, task_id=0)
    parsed = AegisObservation.model_validate(obs.metadata)
    assert parsed.server_snapshots
    assert all(snapshot.server_id.startswith("srv-") for snapshot in parsed.server_snapshots)
    assert "srv-000" not in {snapshot.server_id for snapshot in parsed.server_snapshots}

    target_id = next(
        snapshot.server_id
        for snapshot in parsed.server_snapshots
        if snapshot.hostname.startswith("web-gateway-")
    )
    obs = env.step(ModifyResource(target_server_id=target_id, new_tier="large"))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0
    assert obs.reward == 0.8

    env = AegisEnvironment()
    env.reset(seed=22, task_id=2)
    obs = env.step(DeleteResource(target_server_id="utility-log-rotator", confirm_deletion=True))
    assert obs.done is True
    assert obs.reward == 0.0

    env = AegisEnvironment()
    env.reset(seed=33, task_id=0)
    env.aegis_state.step_count = env.MAX_STEPS - 1
    target_id = env.aegis_state.servers[0].server_id
    obs = env.step(QueryLogs(target_server_id=target_id, severity_filter="info"))
    assert obs.done is True
    assert 0.0 <= obs.reward <= 1.0

    env = AsyncBonusEnvironment()
    env.reset(seed=44, task_id=1)
    obs = asyncio.run(
        env.step_async(
            {
                "action_type": "request_human_confirmation",
                "reasoning_trace": "uncertainty flag indicates high risk of data loss",
                "target_server_id": "",
                "new_tier": "small",
                "confirm_deletion": False,
                "severity_filter": "info",
                "max_entries": 50,
            }
        )
    )
    assert obs.done is True
    assert obs.reward == 0.2
    AegisObservation.model_validate(obs.metadata)

    print("Compliance checklist checks passed.")


if __name__ == "__main__":
    main()
