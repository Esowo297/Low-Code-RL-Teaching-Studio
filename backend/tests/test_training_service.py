from pathlib import Path
import json

from fastapi.testclient import TestClient

from app.api.routes import training_service as routed_training_service
from app.main import app
from app.repositories.experiment_store import ExperimentStore
from app.rl.envs.cliffwalking import CliffWalkingEnv
from app.rl.envs.factory import create_environment
from app.rl.envs.gridworld import GridWorldEnv
from app.rl.envs.windygridworld import WindyGridWorldEnv
from app.schemas.experiment import (
    CliffWalkingConfig,
    DQNConfig,
    DQNExperimentRequest,
    ExperimentResult,
    ExperimentSummary,
    GridEnvironmentView,
    GridWorldConfig,
    QLearningExperimentRequest,
    ReinforceConfig,
    ReinforceExperimentRequest,
    SARSAConfig,
    SARSAExperimentRequest,
    TrainingConfig,
    WindyGridWorldConfig,
)
from app.services.benchmarks import get_benchmark_preset, passes_benchmark_preset
from app.services.training_service import TrainingService


def test_environment_factory_creates_gridworld_env_from_request() -> None:
    request = QLearningExperimentRequest(env_config=GridWorldConfig(size=7))

    env = create_environment(request)

    assert isinstance(env, GridWorldEnv)
    assert env.environment_id == "gridworld"
    assert env.rows == 7
    assert env.cols == 7
    assert env.state_count == 49


def test_environment_factory_creates_cliffwalking_env_from_request() -> None:
    request = QLearningExperimentRequest(
        environment_id="cliffwalking",
        env_config=CliffWalkingConfig(),
    )

    env = create_environment(request)

    assert isinstance(env, CliffWalkingEnv)
    assert env.environment_id == "cliffwalking"
    assert env.rows == 4
    assert env.cols == 12
    assert env.state_count == 48


def test_environment_factory_creates_windygridworld_env_from_request() -> None:
    request = QLearningExperimentRequest(
        environment_id="windygridworld",
        env_config=WindyGridWorldConfig(),
    )

    env = create_environment(request)

    assert isinstance(env, WindyGridWorldEnv)
    assert env.environment_id == "windygridworld"
    assert env.rows == 7
    assert env.cols == 10
    assert env.state_count == 70


def test_gridworld_config_migrates_legacy_size_to_rows_and_cols() -> None:
    config = GridWorldConfig(size=8)
    payload = config.model_dump(mode="json")

    assert config.environment_id == "gridworld"
    assert config.rows == 8
    assert config.cols == 8
    assert config.size == 8
    assert config.goal.row == 7
    assert config.goal.col == 7
    assert config.obstacles == []
    assert config.traps == []
    assert payload["size"] == 8
    assert payload["rows"] == 8
    assert payload["cols"] == 8


def test_cliffwalking_config_builds_default_start_goal_and_cliffs() -> None:
    config = CliffWalkingConfig()

    assert config.environment_id == "cliffwalking"
    assert config.start.row == 3
    assert config.start.col == 0
    assert config.goal.row == 3
    assert config.goal.col == 11
    assert len(config.cliffs) == 10
    assert config.cliffs[0].row == 3
    assert config.cliffs[0].col == 1
    assert config.cliffs[-1].col == 10


def test_windygridworld_config_builds_default_start_goal_and_wind_strengths() -> None:
    config = WindyGridWorldConfig()

    assert config.environment_id == "windygridworld"
    assert config.start.row == 3
    assert config.start.col == 0
    assert config.goal.row == 3
    assert config.goal.col == 7
    assert config.wind_strengths == [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]


def test_experiment_result_populates_environment_view_from_legacy_policy_grid() -> None:
    result = ExperimentResult(
        run_id="run-legacy-grid",
        created_at="2026-05-11T00:00:00+00:00",
        request=QLearningExperimentRequest(),
        summary=ExperimentSummary(
            average_reward=1.0,
            best_reward=2.0,
            success_rate=0.5,
            stable_success_rate=0.6,
        ),
        metrics=[],
        path_traces=[],
        policy_grid=[["START", "R"], ["D", "GOAL"]],
    )
    payload = result.model_dump(mode="json")

    assert result.environment_view is not None
    assert result.environment_view.view_type == "grid"
    assert result.environment_view.rows == 2
    assert result.environment_view.cols == 2
    assert result.environment_view.cells == result.policy_grid
    assert payload["environment_view"]["rows"] == 2
    assert payload["environment_view"]["cells"] == result.policy_grid


def test_experiment_result_populates_legacy_policy_grid_from_environment_view() -> None:
    result = ExperimentResult(
        run_id="run-view-grid",
        created_at="2026-05-11T00:00:00+00:00",
        request=QLearningExperimentRequest(),
        summary=ExperimentSummary(
            average_reward=1.0,
            best_reward=2.0,
            success_rate=0.5,
            stable_success_rate=0.6,
        ),
        metrics=[],
        path_traces=[],
        environment_view=GridEnvironmentView(
            rows=2,
            cols=2,
            cells=[["START", "R"], ["D", "GOAL"]],
        ),
    )

    assert result.policy_grid == [["START", "R"], ["D", "GOAL"]]
    assert result.model_dump(mode="json")["policy_grid"] == [["START", "R"], ["D", "GOAL"]]


def test_training_service_creates_q_learning_result_and_persists_it(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(QLearningExperimentRequest())

    assert result.status == "completed"
    assert len(result.metrics) == result.request.training.episodes
    assert len(result.policy_grid) == result.request.env_config.size
    assert (tmp_path / "experiments.db").exists()
    stored = service.get_run(result.run_id)
    assert stored.run_id == result.run_id


def test_training_service_creates_cliffwalking_q_learning_result(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    request = QLearningExperimentRequest(
        environment_id="cliffwalking",
        env_config=CliffWalkingConfig(),
        training=TrainingConfig(episodes=25, seed=3, trace_frequency=5),
    )

    result = service.run_experiment(request)

    assert result.request.environment_id == "cliffwalking"
    assert len(result.metrics) == 25
    assert len(result.policy_grid) == 4
    assert len(result.policy_grid[0]) == 12
    assert any("CLIFF" in row for row in result.policy_grid)


def test_training_service_creates_windygridworld_q_learning_result(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    request = QLearningExperimentRequest(
        environment_id="windygridworld",
        env_config=WindyGridWorldConfig(),
        training=TrainingConfig(episodes=25, seed=3, trace_frequency=5),
    )

    result = service.run_experiment(request)

    assert result.request.environment_id == "windygridworld"
    assert len(result.metrics) == 25
    assert len(result.policy_grid) == 7
    assert len(result.policy_grid[0]) == 10
    assert result.policy_grid[3][0] == "START"
    assert result.policy_grid[3][7] == "GOAL"


def test_cliffwalking_result_does_not_match_gridworld_benchmark(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(
        QLearningExperimentRequest(
            environment_id="cliffwalking",
            env_config=CliffWalkingConfig(),
            training=TrainingConfig(episodes=20, seed=4, trace_frequency=5),
        )
    )
    benchmark = get_benchmark_preset("teacher_gridworld_q_learning_baseline")

    assert passes_benchmark_preset(result, benchmark) is False


def test_cliffwalking_result_does_not_match_cliffwalking_benchmark_with_different_shape(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(
        QLearningExperimentRequest(
            environment_id="cliffwalking",
            env_config=CliffWalkingConfig(rows=5, cols=12),
            training=TrainingConfig(episodes=20, seed=4, trace_frequency=5),
        )
    )
    benchmark = get_benchmark_preset("teacher_cliffwalking_q_learning_baseline")

    assert passes_benchmark_preset(result, benchmark) is False


def test_windygridworld_result_does_not_match_benchmark_with_different_wind_strengths(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(
        QLearningExperimentRequest(
            environment_id="windygridworld",
            env_config=WindyGridWorldConfig(wind_strengths=[0, 0, 1, 1, 1, 2, 2, 1, 0, 0]),
            training=TrainingConfig(episodes=20, seed=4, trace_frequency=5),
        )
    )
    benchmark = get_benchmark_preset("teacher_windygridworld_q_learning_baseline")

    assert passes_benchmark_preset(result, benchmark) is False


def test_training_service_creates_dqn_result(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    request = DQNExperimentRequest(
        training=TrainingConfig(episodes=30, seed=5, trace_frequency=10),
        algorithm_config=DQNConfig(
            learning_rate=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.98,
            max_steps_per_episode=50,
            batch_size=16,
            replay_buffer_size=300,
            target_sync_interval=10,
            warmup_steps=16,
            hidden_dim=32,
        ),
    )

    result = service.run_experiment(request)

    assert result.request.algorithm_id == "dqn"
    assert len(result.metrics) == 30
    assert len(result.policy_grid) == result.request.env_config.size


def test_training_service_creates_reinforce_result(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    request = ReinforceExperimentRequest(
        training=TrainingConfig(episodes=30, seed=5, trace_frequency=10),
        algorithm_config=ReinforceConfig(
            learning_rate=0.01,
            gamma=0.95,
            max_steps_per_episode=50,
            hidden_dim=32,
        ),
    )

    result = service.run_experiment(request)

    assert result.request.algorithm_id == "reinforce"
    assert len(result.metrics) == 30
    assert len(result.policy_grid) == result.request.env_config.size


def test_training_service_creates_sarsa_result(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    request = SARSAExperimentRequest(
        training=TrainingConfig(episodes=30, seed=5, trace_frequency=10),
        algorithm_config=SARSAConfig(
            learning_rate=0.2,
            gamma=0.92,
            epsilon_start=1.0,
            epsilon_min=0.05,
            epsilon_decay=0.98,
            max_steps_per_episode=50,
        ),
    )

    result = service.run_experiment(request)

    assert result.request.algorithm_id == "sarsa"
    assert len(result.metrics) == 30
    assert len(result.policy_grid) == result.request.env_config.size


def test_experiment_stream_websocket_emits_progress_and_completion() -> None:
    client = TestClient(app)
    payload = {
        "name": "Streaming Q-Learning Demo",
        "environment_id": "gridworld",
        "algorithm_id": "q_learning",
        "persist_result": False,
        "env_config": {
            "size": 6,
            "start": {"row": 0, "col": 0},
            "goal": {"row": 5, "col": 5},
            "obstacles": [],
            "traps": [],
            "rewards": {
                "step_penalty": -1,
                "goal_reward": 20,
                "wall_penalty": -3,
                "trap_penalty": -10,
            },
        },
        "algorithm_config": {
            "learning_rate": 0.2,
            "gamma": 0.92,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.98,
            "max_steps_per_episode": 50,
        },
        "training": {
            "episodes": 20,
            "seed": 4,
            "trace_frequency": 5,
        },
    }

    with client.websocket_connect("/api/experiments/stream") as websocket:
        websocket.send_json(payload)

        started_event = websocket.receive_json()
        assert started_event["type"] == "started"

        metric_events = 0
        completed_event = None
        while True:
            event = websocket.receive_json()
            if event["type"] == "metric":
                metric_events += 1
                continue
            if event["type"] == "completed":
                completed_event = event
                break

        assert metric_events == 20
        assert completed_event is not None
        assert completed_event["result"]["request"]["algorithm_id"] == "q_learning"


def test_experiment_stream_supports_pause_resume_and_cancel() -> None:
    client = TestClient(app)
    payload = {
        "name": "Streaming DQN Demo",
        "environment_id": "gridworld",
        "algorithm_id": "dqn",
        "persist_result": False,
        "env_config": {
            "size": 6,
            "start": {"row": 0, "col": 0},
            "goal": {"row": 5, "col": 5},
            "obstacles": [],
            "traps": [],
            "rewards": {
                "step_penalty": -1,
                "goal_reward": 20,
                "wall_penalty": -3,
                "trap_penalty": -10,
            },
        },
        "algorithm_config": {
            "learning_rate": 0.001,
            "gamma": 0.95,
            "epsilon_start": 1.0,
            "epsilon_min": 0.05,
            "epsilon_decay": 0.99,
            "max_steps_per_episode": 30,
            "batch_size": 16,
            "replay_buffer_size": 300,
            "target_sync_interval": 10,
            "warmup_steps": 16,
            "hidden_dim": 32,
        },
        "training": {
            "episodes": 80,
            "seed": 9,
            "trace_frequency": 10,
        },
    }

    def receive_until(websocket, accepted_types: set[str]) -> dict:
        for _ in range(400):
            event = websocket.receive_json()
            if event["type"] in accepted_types:
                return event
        raise AssertionError(f"Did not receive any of {accepted_types}")

    with client.websocket_connect("/api/experiments/stream") as websocket:
        websocket.send_json(payload)

        started_event = receive_until(websocket, {"started"})
        assert started_event["type"] == "started"

        first_metric = receive_until(websocket, {"metric"})
        assert first_metric["type"] == "metric"

        websocket.send_json({"type": "pause"})
        paused_event = receive_until(websocket, {"control"})
        assert paused_event["state"] == "paused"

        websocket.send_json({"type": "resume"})
        resumed_event = receive_until(websocket, {"control"})
        assert resumed_event["state"] == "running"

        websocket.send_json({"type": "cancel"})
        cancelling_event = receive_until(websocket, {"control"})
        assert cancelling_event["state"] == "cancelling"

        final_event = receive_until(websocket, {"cancelled", "completed"})
        assert final_event["type"] == "cancelled"


def test_get_experiment_endpoint_returns_saved_result(tmp_path: Path) -> None:
    original_store = routed_training_service.store
    routed_training_service.store = ExperimentStore(base_path=tmp_path)

    try:
        result = routed_training_service.run_experiment(QLearningExperimentRequest())
        client = TestClient(app)
        response = client.get(f"/api/experiments/{result.run_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["run_id"] == result.run_id
        assert payload["request"]["algorithm_id"] == "q_learning"
    finally:
        routed_training_service.store = original_store


def test_get_benchmark_catalog_returns_teacher_presets() -> None:
    client = TestClient(app)

    response = client.get("/api/benchmarks")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["benchmarks"]) >= 10
    benchmark_ids = {benchmark["id"] for benchmark in payload["benchmarks"]}
    assert {
        "teacher_gridworld_q_learning_baseline",
        "teacher_gridworld_sarsa_baseline",
        "teacher_gridworld_dqn_baseline",
        "teacher_gridworld_reinforce_baseline",
        "teacher_cliffwalking_q_learning_baseline",
        "teacher_cliffwalking_sarsa_baseline",
        "teacher_cliffwalking_dqn_baseline",
        "teacher_windygridworld_q_learning_baseline",
        "teacher_windygridworld_sarsa_baseline",
        "teacher_windygridworld_dqn_baseline",
    }.issubset(benchmark_ids)
    assert payload["benchmarks"][0]["request"]["algorithm_id"] == "q_learning"
    assert payload["benchmarks"][1]["request"]["algorithm_id"] == "sarsa"
    assert payload["benchmarks"][2]["request"]["algorithm_id"] == "dqn"
    assert payload["benchmarks"][3]["request"]["algorithm_id"] == "reinforce"
    assert payload["benchmarks"][0]["thresholds"][0]["metric_id"] == "average_reward"
    assert payload["benchmarks"][0]["request"]["submission_role"] == "teacher"
    cliff_benchmarks = [benchmark for benchmark in payload["benchmarks"] if benchmark["request"]["environment_id"] == "cliffwalking"]
    assert {benchmark["request"]["algorithm_id"] for benchmark in cliff_benchmarks} == {"q_learning", "sarsa", "dqn"}
    windy_benchmarks = [
        benchmark for benchmark in payload["benchmarks"] if benchmark["request"]["environment_id"] == "windygridworld"
    ]
    assert {benchmark["request"]["algorithm_id"] for benchmark in windy_benchmarks} == {"q_learning", "sarsa", "dqn"}


def test_get_catalog_exposes_cliffwalking_environment() -> None:
    client = TestClient(app)

    response = client.get("/api/catalog")

    assert response.status_code == 200
    payload = response.json()
    environment_ids = {item["id"] for item in payload["environments"]}
    assert {"gridworld", "cliffwalking", "windygridworld"}.issubset(environment_ids)


def test_get_assignment_catalog_returns_builtin_templates() -> None:
    client = TestClient(app)

    response = client.get("/api/assignments")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["assignments"]) >= 10
    assignment_ids = {assignment["id"] for assignment in payload["assignments"]}
    assert {
        "assignment_gridworld_q_learning_convergence",
        "assignment_gridworld_sarsa_policy_comparison",
        "assignment_gridworld_dqn_comparison",
        "assignment_gridworld_reinforce_policy_gradient",
        "assignment_cliffwalking_q_learning_risk_path",
        "assignment_cliffwalking_sarsa_safe_policy",
        "assignment_cliffwalking_dqn_value_approximation",
        "assignment_windygridworld_q_learning_wind_compensation",
        "assignment_windygridworld_sarsa_online_correction",
        "assignment_windygridworld_dqn_value_learning",
    }.issubset(assignment_ids)
    assert payload["assignments"][0]["benchmark_id"] == "teacher_gridworld_q_learning_baseline"
    assert payload["assignments"][1]["request"]["algorithm_id"] == "sarsa"
    assert payload["assignments"][2]["request"]["assignment_id"] == "assignment_gridworld_dqn_comparison"
    assert payload["assignments"][3]["request"]["algorithm_id"] == "reinforce"
    cliff_assignments = [assignment for assignment in payload["assignments"] if assignment["request"]["environment_id"] == "cliffwalking"]
    assert {assignment["request"]["algorithm_id"] for assignment in cliff_assignments} == {"q_learning", "sarsa", "dqn"}
    windy_assignments = [
        assignment for assignment in payload["assignments"] if assignment["request"]["environment_id"] == "windygridworld"
    ]
    assert {assignment["request"]["algorithm_id"] for assignment in windy_assignments} == {"q_learning", "sarsa", "dqn"}


def test_render_report_endpoint_returns_markdown_with_benchmark_section(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(QLearningExperimentRequest())
    client = TestClient(app)

    response = client.post(
        "/api/reports/render",
        json={
            "result": result.model_dump(mode="json"),
            "benchmark_id": "teacher_gridworld_q_learning_baseline",
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert result.run_id in response.text
    assert "教师基准评估" in response.text
    assert "Q-Learning" in response.text


def test_render_report_endpoint_supports_cliffwalking_configuration(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(
        QLearningExperimentRequest(
            environment_id="cliffwalking",
            env_config=CliffWalkingConfig(),
            training=TrainingConfig(episodes=25, seed=6, trace_frequency=5),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/reports/render",
        json={
            "result": result.model_dump(mode="json"),
            "benchmark_id": "teacher_cliffwalking_q_learning_baseline",
        },
    )

    assert response.status_code == 200
    assert "CliffWalking" in response.text
    assert "悬崖单元" in response.text
    assert "陷阱单元" not in response.text


def test_render_report_endpoint_supports_windygridworld_configuration(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(
        QLearningExperimentRequest(
            environment_id="windygridworld",
            env_config=WindyGridWorldConfig(),
            training=TrainingConfig(episodes=25, seed=6, trace_frequency=5),
        )
    )
    client = TestClient(app)

    response = client.post(
        "/api/reports/render",
        json={
            "result": result.model_dump(mode="json"),
            "benchmark_id": "teacher_windygridworld_q_learning_baseline",
        },
    )

    assert response.status_code == 200
    assert "WindyGridWorld" in response.text
    assert "列风强度" in response.text
    assert "悬崖单元" not in response.text


def test_classroom_analytics_endpoint_aggregates_saved_runs(tmp_path: Path) -> None:
    original_store = routed_training_service.store
    routed_training_service.store = ExperimentStore(base_path=tmp_path)

    try:
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Alice",
                submission_role="student",
                assignment_id="assignment_gridworld_q_learning_convergence",
                assignment_title="Assignment 1 | Q-Learning Convergence in GridWorld",
                training=TrainingConfig(episodes=20, seed=11, trace_frequency=5),
            )
        )
        routed_training_service.run_experiment(
            DQNExperimentRequest(
                submitted_by="Bob",
                submission_role="student",
                assignment_id="assignment_gridworld_dqn_comparison",
                assignment_title="Assignment 2 | DQN Stability and Comparison",
                training=TrainingConfig(episodes=20, seed=12, trace_frequency=5),
                algorithm_config=DQNConfig(
                    learning_rate=0.001,
                    gamma=0.95,
                    epsilon_start=1.0,
                    epsilon_min=0.05,
                    epsilon_decay=0.99,
                    max_steps_per_episode=30,
                    batch_size=16,
                    replay_buffer_size=300,
                    target_sync_interval=10,
                    warmup_steps=16,
                    hidden_dim=32,
                ),
            )
        )
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Instructor",
                submission_role="teacher",
                persist_result=True,
                assignment_id="assignment_gridworld_q_learning_convergence",
                assignment_title="Assignment 1 | Q-Learning Convergence in GridWorld",
                training=TrainingConfig(episodes=20, seed=13, trace_frequency=5),
            )
        )

        client = TestClient(app)
        response = client.get(
            "/api/analytics/classroom",
            params={
                "limit": 10,
                "benchmark_id": "teacher_gridworld_q_learning_baseline",
                "assignment_id": "assignment_gridworld_q_learning_convergence",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["assignment_filter_id"] == "assignment_gridworld_q_learning_convergence"
        assert payload["total_runs"] == 2
        assert payload["student_runs"] == 1
        assert payload["teacher_runs"] == 1
        assert payload["distinct_submitters"] == 2
        assert payload["benchmark"]["benchmark_id"] == "teacher_gridworld_q_learning_baseline"
        assert payload["benchmark"]["evaluated_runs"] == 1
        assert payload["assignments"][0]["assignment_id"] == "assignment_gridworld_q_learning_convergence"
        assert {entry["submitted_by"] for entry in payload["students"]} == {"Alice"}
    finally:
        routed_training_service.store = original_store


def test_classroom_analytics_ignores_shape_mismatched_runs_for_cliff_benchmark(tmp_path: Path) -> None:
    original_store = routed_training_service.store
    routed_training_service.store = ExperimentStore(base_path=tmp_path)

    try:
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Alice",
                submission_role="student",
                environment_id="cliffwalking",
                env_config=CliffWalkingConfig(),
                training=TrainingConfig(episodes=20, seed=21, trace_frequency=5),
            )
        )
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Bob",
                submission_role="student",
                environment_id="cliffwalking",
                env_config=CliffWalkingConfig(rows=5, cols=12),
                training=TrainingConfig(episodes=20, seed=22, trace_frequency=5),
            )
        )

        client = TestClient(app)
        response = client.get(
            "/api/analytics/classroom",
            params={
                "limit": 10,
                "benchmark_id": "teacher_cliffwalking_q_learning_baseline",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_runs"] == 2
        assert payload["benchmark"]["benchmark_id"] == "teacher_cliffwalking_q_learning_baseline"
        assert payload["benchmark"]["evaluated_runs"] == 1
    finally:
        routed_training_service.store = original_store


def test_classroom_analytics_ignores_config_mismatched_runs_for_windy_benchmark(tmp_path: Path) -> None:
    original_store = routed_training_service.store
    routed_training_service.store = ExperimentStore(base_path=tmp_path)

    try:
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Alice",
                submission_role="student",
                environment_id="windygridworld",
                env_config=WindyGridWorldConfig(),
                training=TrainingConfig(episodes=20, seed=31, trace_frequency=5),
            )
        )
        routed_training_service.run_experiment(
            QLearningExperimentRequest(
                submitted_by="Bob",
                submission_role="student",
                environment_id="windygridworld",
                env_config=WindyGridWorldConfig(wind_strengths=[0, 0, 1, 1, 1, 2, 2, 1, 0, 0]),
                training=TrainingConfig(episodes=20, seed=32, trace_frequency=5),
            )
        )

        client = TestClient(app)
        response = client.get(
            "/api/analytics/classroom",
            params={
                "limit": 10,
                "benchmark_id": "teacher_windygridworld_q_learning_baseline",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["total_runs"] == 2
        assert payload["benchmark"]["benchmark_id"] == "teacher_windygridworld_q_learning_baseline"
        assert payload["benchmark"]["evaluated_runs"] == 1
    finally:
        routed_training_service.store = original_store


def test_store_imports_legacy_json_runs_into_sqlite(tmp_path: Path) -> None:
    legacy_dir = tmp_path / "experiments"
    legacy_dir.mkdir(parents=True, exist_ok=True)

    result = TrainingService(store=ExperimentStore(base_path=tmp_path / "seed")).run_experiment(QLearningExperimentRequest())
    legacy_payload = result.model_dump(mode="json")
    (legacy_dir / f"{result.run_id}.json").write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")

    migrated_store = ExperimentStore(base_path=tmp_path)
    history = migrated_store.list_runs(limit=5)

    assert any(entry.run_id == result.run_id for entry in history)
    loaded = migrated_store.load(result.run_id)
    assert loaded.request.algorithm_id == "q_learning"
