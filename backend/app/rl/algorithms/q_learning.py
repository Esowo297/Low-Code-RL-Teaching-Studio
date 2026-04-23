from __future__ import annotations

import asyncio

import numpy as np

from app.rl.envs.gridworld import ACTION_CODES, GridWorldEnv
from app.rl.training_artifacts import AsyncProgressCallback, ProgressUpdate, StreamController, TrainingArtifacts
from app.schemas.experiment import EpisodeMetric, PathTrace, QLearningConfig, TrainingConfig


class QLearningTrainer:
    def __init__(
        self,
        env: GridWorldEnv,
        algorithm_config: QLearningConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.env = env
        self.algorithm_config = algorithm_config
        self.training_config = training_config
        self.rng = np.random.default_rng(training_config.seed)

    def train(self) -> TrainingArtifacts:
        return asyncio.run(self._train())

    async def train_stream(
        self,
        on_progress: AsyncProgressCallback,
        controller: StreamController | None = None,
    ) -> TrainingArtifacts:
        return await self._train(on_progress=on_progress, controller=controller)

    async def _train(
        self,
        on_progress: AsyncProgressCallback | None = None,
        controller: StreamController | None = None,
    ) -> TrainingArtifacts:
        q_table = np.zeros((self.env.size * self.env.size, len(ACTION_CODES)), dtype=float)
        metrics: list[EpisodeMetric] = []
        path_traces: list[PathTrace] = []
        epsilon = self.algorithm_config.epsilon_start
        success_flags: list[int] = []
        cumulative_reward = 0.0

        for episode in range(1, self.training_config.episodes + 1):
            if controller is not None:
                await controller.checkpoint()

            state = self.env.reset()
            total_reward = 0.0
            td_errors: list[float] = []
            episode_path = [state.model_copy(deep=True)]
            success = False

            for _ in range(self.algorithm_config.max_steps_per_episode):
                if controller is not None:
                    await controller.checkpoint()

                action = self._select_action(q_table, state, epsilon)
                transition = self.env.step(action)

                state_index = self.env.state_to_index(state)
                next_index = self.env.state_to_index(transition.state)

                target = transition.reward
                if not transition.done:
                    target += self.algorithm_config.gamma * float(np.max(q_table[next_index]))

                td_error = target - q_table[state_index, action]
                q_table[state_index, action] += self.algorithm_config.learning_rate * td_error

                total_reward += transition.reward
                td_errors.append(abs(float(td_error)))
                state = transition.state
                episode_path.append(state.model_copy(deep=True))

                if transition.done:
                    success = transition.terminated_by == "goal"
                    break

            epsilon = max(self.algorithm_config.epsilon_min, epsilon * self.algorithm_config.epsilon_decay)
            success_flags.append(1 if success else 0)
            cumulative_reward += total_reward

            metric = EpisodeMetric(
                episode=episode,
                reward=round(total_reward, 4),
                td_error=round(float(np.mean(td_errors)) if td_errors else 0.0, 6),
                epsilon=round(epsilon, 6),
                steps=len(episode_path) - 1,
                success=success,
            )
            metrics.append(metric)

            latest_trace: PathTrace | None = None
            if episode % self.training_config.trace_frequency == 0 or episode == self.training_config.episodes:
                latest_trace = PathTrace(
                    episode=episode,
                    success=success,
                    total_reward=round(total_reward, 4),
                    path=episode_path,
                )
                path_traces.append(latest_trace)

            if on_progress is not None:
                await on_progress(
                    ProgressUpdate(
                        metric=metric,
                        latest_trace=latest_trace,
                        completed_episodes=episode,
                        total_episodes=self.training_config.episodes,
                        running_average_reward=round(cumulative_reward / episode, 4),
                        running_success_rate=round(float(np.mean(success_flags)), 4),
                    )
                )

        reward_values = [metric.reward for metric in metrics]
        average_reward = float(np.mean(reward_values)) if reward_values else 0.0
        success_rate = float(np.mean(success_flags)) if success_flags else 0.0
        rolling_window = min(20, len(success_flags))
        stable_success_rate = float(np.mean(success_flags[-rolling_window:])) if rolling_window else 0.0

        return TrainingArtifacts(
            metrics=metrics,
            path_traces=path_traces,
            policy_grid=self._build_policy_grid(q_table),
            average_reward=round(average_reward, 4),
            best_reward=round(float(np.max(reward_values)) if reward_values else 0.0, 4),
            success_rate=round(success_rate, 4),
            stable_success_rate=round(stable_success_rate, 4),
        )

    def _select_action(self, q_table: np.ndarray, state, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, len(ACTION_CODES)))
        state_index = self.env.state_to_index(state)
        return int(np.argmax(q_table[state_index]))

    def _build_policy_grid(self, q_table: np.ndarray) -> list[list[str]]:
        grid: list[list[str]] = []
        for row in range(self.env.size):
            current_row: list[str] = []
            for col in range(self.env.size):
                special_token = self.env.cell_token(row, col)
                if special_token is not None:
                    current_row.append(special_token)
                    continue

                state_index = row * self.env.size + col
                action_index = int(np.argmax(q_table[state_index]))
                current_row.append(ACTION_CODES[action_index])
            grid.append(current_row)
        return grid
