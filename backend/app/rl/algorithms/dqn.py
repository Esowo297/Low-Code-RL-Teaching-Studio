from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from app.rl.envs.base import ACTION_CODES, DiscreteGridEnvironment
from app.rl.training_artifacts import AsyncProgressCallback, ProgressUpdate, StreamController, TrainingArtifacts
from app.schemas.experiment import DQNConfig, EpisodeMetric, PathTrace, TrainingConfig


@dataclass(slots=True)
class ReplayTransition:
    state_index: int
    action: int
    reward: float
    next_state_index: int
    done: bool


class DQNNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class DQNTrainer:
    def __init__(
        self,
        env: DiscreteGridEnvironment,
        algorithm_config: DQNConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.env = env
        self.algorithm_config = algorithm_config
        self.training_config = training_config
        self.rng = np.random.default_rng(training_config.seed)
        self.device = torch.device("cpu")
        torch.manual_seed(training_config.seed)

        self.state_dim = self.env.state_count
        self.action_dim = len(ACTION_CODES)
        self.policy_net = DQNNetwork(self.state_dim, algorithm_config.hidden_dim, self.action_dim).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, algorithm_config.hidden_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=algorithm_config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer: deque[ReplayTransition] = deque(maxlen=algorithm_config.replay_buffer_size)
        self.optimization_steps = 0

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
            loss_values: list[float] = []
            episode_path = [state.model_copy(deep=True)]
            success = False

            for _ in range(self.algorithm_config.max_steps_per_episode):
                if controller is not None:
                    await controller.checkpoint()

                action = self._select_action(self.env.state_to_index(state), epsilon)
                transition = self.env.step(action)
                next_state = transition.state

                self.replay_buffer.append(
                    ReplayTransition(
                        state_index=self.env.state_to_index(state),
                        action=action,
                        reward=transition.reward,
                        next_state_index=self.env.state_to_index(next_state),
                        done=transition.done,
                    )
                )

                if len(self.replay_buffer) >= self.algorithm_config.batch_size and len(self.replay_buffer) >= self.algorithm_config.warmup_steps:
                    loss = self._optimize_model()
                    if loss is not None:
                        loss_values.append(loss)

                total_reward += transition.reward
                state = next_state
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
                td_error=round(float(np.mean(loss_values)) if loss_values else 0.0, 6),
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
            policy_grid=self._build_policy_grid(),
            average_reward=round(average_reward, 4),
            best_reward=round(float(np.max(reward_values)) if reward_values else 0.0, 4),
            success_rate=round(success_rate, 4),
            stable_success_rate=round(stable_success_rate, 4),
        )

    def _select_action(self, state_index: int, epsilon: float) -> int:
        if self.rng.random() < epsilon:
            return int(self.rng.integers(0, self.action_dim))

        with torch.no_grad():
            q_values = self.policy_net(self._state_batch([state_index]))
            return int(torch.argmax(q_values, dim=1).item())

    def _optimize_model(self) -> float | None:
        if len(self.replay_buffer) < self.algorithm_config.batch_size:
            return None

        sample_indices = self.rng.choice(len(self.replay_buffer), size=self.algorithm_config.batch_size, replace=False)
        transitions = [self.replay_buffer[index] for index in sample_indices]

        state_batch = self._state_batch([transition.state_index for transition in transitions])
        next_state_batch = self._state_batch([transition.next_state_index for transition in transitions])
        action_tensor = torch.tensor([transition.action for transition in transitions], dtype=torch.long, device=self.device)
        reward_tensor = torch.tensor([transition.reward for transition in transitions], dtype=torch.float32, device=self.device)
        done_tensor = torch.tensor([transition.done for transition in transitions], dtype=torch.float32, device=self.device)

        current_q = self.policy_net(state_batch).gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q = self.target_net(next_state_batch).max(dim=1).values
            target_q = reward_tensor + self.algorithm_config.gamma * next_q * (1.0 - done_tensor)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.algorithm_config.target_sync_interval == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def _state_batch(self, state_indices: list[int]) -> torch.Tensor:
        index_tensor = torch.tensor(state_indices, dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(index_tensor, num_classes=self.state_dim).to(torch.float32)

    def _build_policy_grid(self) -> list[list[str]]:
        grid: list[list[str]] = []
        for row in range(self.env.rows):
            current_row: list[str] = []
            for col in range(self.env.cols):
                special_token = self.env.cell_token(row, col)
                if special_token is not None:
                    current_row.append(special_token)
                    continue

                state_index = row * self.env.cols + col
                with torch.no_grad():
                    q_values = self.policy_net(self._state_batch([state_index]))
                    action_index = int(torch.argmax(q_values, dim=1).item())
                current_row.append(ACTION_CODES[action_index])
            grid.append(current_row)
        return grid
