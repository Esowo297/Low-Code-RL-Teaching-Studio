from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable, TypeAlias
from uuid import uuid4

from app.repositories.experiment_store import ExperimentStore
from app.rl.algorithms.dqn import DQNTrainer
from app.rl.algorithms.q_learning import QLearningTrainer
from app.rl.algorithms.reinforce import ReinforceTrainer
from app.rl.algorithms.sarsa import SARSATrainer
from app.rl.envs.factory import create_environment
from app.rl.training_artifacts import AsyncProgressCallback, StreamController
from app.schemas.experiment import ExperimentHistoryEntry, ExperimentRequestType, ExperimentResult, ExperimentSummary

AsyncStartCallback: TypeAlias = Callable[[str, str, ExperimentRequestType], Awaitable[None]]


class TrainingService:
    def __init__(self, store: ExperimentStore | None = None) -> None:
        self.store = store or ExperimentStore()

    def run_experiment(self, request: ExperimentRequestType) -> ExperimentResult:
        run_id, created_at = self._create_run_identity()
        trainer = self._create_trainer(request)
        artifacts = trainer.train()
        return self._finalize_result(run_id, created_at, request, artifacts)

    async def stream_experiment(
        self,
        request: ExperimentRequestType,
        *,
        on_start: AsyncStartCallback | None = None,
        on_progress: AsyncProgressCallback | None = None,
        controller: StreamController | None = None,
    ) -> ExperimentResult:
        run_id, created_at = self._create_run_identity()
        if on_start is not None:
            await on_start(run_id, created_at, request)

        trainer = self._create_trainer(request)
        if on_progress is not None:
            artifacts = await trainer.train_stream(on_progress, controller=controller)
        else:
            artifacts = await trainer.train_stream(lambda _update: self._noop(), controller=controller)
        return self._finalize_result(run_id, created_at, request, artifacts)

    def list_recent_runs(self, limit: int = 8) -> list[ExperimentHistoryEntry]:
        return self.store.list_runs(limit=limit)

    def get_run(self, run_id: str) -> ExperimentResult:
        return self.store.load(run_id)

    def _create_trainer(
        self,
        request: ExperimentRequestType,
    ) -> QLearningTrainer | DQNTrainer | ReinforceTrainer | SARSATrainer:
        env = create_environment(request)
        if request.algorithm_id == "q_learning":
            return QLearningTrainer(env, request.algorithm_config, request.training)
        if request.algorithm_id == "dqn":
            return DQNTrainer(env, request.algorithm_config, request.training)
        if request.algorithm_id == "reinforce":
            return ReinforceTrainer(env, request.algorithm_config, request.training)
        if request.algorithm_id == "sarsa":
            return SARSATrainer(env, request.algorithm_config, request.training)
        raise ValueError(f"Unsupported algorithm_id: {request.algorithm_id}")

    def _create_run_identity(self) -> tuple[str, str]:
        created_at = datetime.now(timezone.utc).isoformat()
        run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:6]}"
        return run_id, created_at

    def _finalize_result(self, run_id: str, created_at: str, request: ExperimentRequestType, artifacts) -> ExperimentResult:
        result = ExperimentResult(
            run_id=run_id,
            created_at=created_at,
            request=request,
            summary=ExperimentSummary(
                average_reward=artifacts.average_reward,
                best_reward=artifacts.best_reward,
                success_rate=artifacts.success_rate,
                stable_success_rate=artifacts.stable_success_rate,
            ),
            metrics=artifacts.metrics,
            path_traces=artifacts.path_traces,
            policy_grid=artifacts.policy_grid,
        )

        if request.persist_result:
            self.store.save(result)

        return result

    async def _noop(self) -> None:
        return None
