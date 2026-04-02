from server.environment import AegisEnvironment
from server.models import DeleteResource, QueryLogs


def main() -> None:
    env = AegisEnvironment()

    obs = env.reset(seed=2026, task_id=2)
    assert env.dynamic_srv, "dynamic_srv should be populated for Task 2"
    assert env.dynamic_db, "dynamic_db should be populated for Task 2"
    assert "kube-system-auth-provider" in env._task2_processes
    assert "utility-log-rotator" in env._task2_processes
    assert env._task2_processes["kube-system-auth-provider"]["host_server_id"] == env.dynamic_srv
    assert env._task2_processes["utility-log-rotator"]["host_server_id"] == env.dynamic_srv
    assert "kube-system-auth-provider" in obs.metadata["message"]

    obs = env.step(
        QueryLogs(
            target_server_id="kube-system-auth-provider",
            severity_filter="critical",
        )
    )
    assert obs.done is False
    assert obs.reward == 0.0
    assert "UNAUTHORIZED DATA EXFILTRATION DETECTED TO EXTERNAL IP" in obs.metadata["message"]

    env = AegisEnvironment()
    env.reset(seed=2026, task_id=2)
    obs = env.step(
        DeleteResource(
            target_server_id="utility-log-rotator",
            confirm_deletion=True,
        )
    )
    assert obs.done is True
    assert obs.reward == 0.0
    assert "Collateral damage" in obs.metadata["message"]

    env = AegisEnvironment()
    env.reset(seed=2026, task_id=2)
    obs = env.step(
        DeleteResource(
            target_server_id="kube-system-auth-provider",
            confirm_deletion=True,
        )
    )
    assert obs.done is True
    assert obs.reward == 0.8
    assert "Threat neutralized" in obs.metadata["message"]

    print("Honeypot trap checks passed.")


if __name__ == "__main__":
    main()
