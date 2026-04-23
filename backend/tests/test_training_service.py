from pathlib import Path
import json

from fastapi.testclient import TestClient

from app.api.routes import training_service as routed_training_service
from app.main import app
from app.repositories.experiment_store import ExperimentStore
from app.schemas.experiment import DQNConfig, DQNExperimentRequest, QLearningExperimentRequest, TrainingConfig
from app.services.training_service import TrainingService


def test_training_service_creates_q_learning_result_and_persists_it(tmp_path: Path) -> None:
    service = TrainingService(store=ExperimentStore(base_path=tmp_path))
    result = service.run_experiment(QLearningExperimentRequest())

    assert result.status == "completed"
    assert len(result.metrics) == result.request.training.episodes
    assert len(result.policy_grid) == result.request.env_config.size
    assert (tmp_path / "experiments.db").exists()
    stored = service.get_run(result.run_id)
    assert stored.run_id == result.run_id


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
    assert len(payload["benchmarks"]) >= 2
    assert payload["benchmarks"][0]["request"]["algorithm_id"] == "q_learning"
    assert payload["benchmarks"][1]["request"]["algorithm_id"] == "dqn"
    assert payload["benchmarks"][0]["thresholds"][0]["metric_id"] == "average_reward"
    assert payload["benchmarks"][0]["request"]["submission_role"] == "teacher"


def test_get_assignment_catalog_returns_builtin_templates() -> None:
    client = TestClient(app)

    response = client.get("/api/assignments")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["assignments"]) >= 2
    assert payload["assignments"][0]["benchmark_id"] == "teacher_gridworld_q_learning_baseline"
    assert payload["assignments"][1]["request"]["assignment_id"] == "assignment_gridworld_dqn_comparison"


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
    assert "Teacher Benchmark Evaluation" in response.text
    assert "Teacher Baseline | Q-Learning Fundamentals" in response.text


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
