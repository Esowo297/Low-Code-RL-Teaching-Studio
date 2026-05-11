from __future__ import annotations

import asyncio

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from app.rl.envs.gridworld import ACTION_CODES, GridWorldEnv
from app.rl.training_artifacts import AsyncProgressCallback, ProgressUpdate, StreamController, TrainingArtifacts
from app.schemas.experiment import EpisodeMetric, PathTrace, ReinforceConfig, TrainingConfig


class ReinforcePolicyNetwork(nn.Module):
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


class ReinforceTrainer:
    def __init__(
        self,
        env: GridWorldEnv,
        algorithm_config: ReinforceConfig,
        training_config: TrainingConfig,
    ) -> None:
        self.env = env
        self.algorithm_config = algorithm_config
        self.training_config = training_config
        self.device = torch.device("cpu")
        self.action_dim = len(ACTION_CODES)
        self.state_dim = env.size * env.size
        self.rng = np.random.default_rng(training_config.seed)

        torch.manual_seed(training_config.seed)
        self.policy_net = ReinforcePolicyNetwork(self.state_dim, algorithm_config.hidden_dim, self.action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=algorithm_config.learning_rate)

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
        success_flags: list[int] = []
        cumulative_reward = 0.0

        for episode in range(1, self.training_config.episodes + 1):
            if controller is not None:
                await controller.checkpoint()

            state = self.env.reset()
            log_probs: list[torch.Tensor] = []
            rewards: list[float] = []
            entropies: list[float] = []
            episode_path = [state.model_copy(deep=True)]
            total_reward = 0.0
            success = False

            for _ in range(self.algorithm_config.max_steps_per_episode):
                if controller is not None:
                    await controller.checkpoint()

                state_index = self.env.state_to_index(state)
                logits = self.policy_net(self._state_batch([state_index])).squeeze(0)
                distribution = Categorical(logits=logits)
                action_tensor = distribution.sample()
                action = int(action_tensor.item())

                log_probs.append(distribution.log_prob(action_tensor))
                entropies.append(float(distribution.entropy().item()))

                transition = self.env.step(action)
                total_reward += transition.reward
                rewards.append(transition.reward)
                state = transition.state
                episode_path.append(state.model_copy(deep=True))

                if transition.done:
                    success = transition.terminated_by == "goal"
                    break

            policy_loss = self._optimize_episode(log_probs, rewards)
            entropy_score = 0.0
            if entropies:
                max_entropy = float(np.log(self.action_dim)) if self.action_dim > 1 else 1.0
                entropy_score = float(np.mean(entropies) / max_entropy) if max_entropy > 0 else 0.0

            success_flags.append(1 if success else 0)
            cumulative_reward += total_reward

            metric = EpisodeMetric(
                episode=episode,
                reward=round(total_reward, 4),
                td_error=round(policy_loss, 6),
                epsilon=round(entropy_score, 6),
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

    def _optimize_episode(self, log_probs: list[torch.Tensor], rewards: list[float]) -> float:
        if not log_probs:
            return 0.0

        returns = self._discounted_returns(rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if len(returns) > 1:
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std(unbiased=False) + 1e-8)

        log_prob_tensor = torch.stack(log_probs)
        loss = -(log_prob_tensor * returns_tensor).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.detach().abs().item())

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        running_return = 0.0
        returns: list[float] = []
        for reward in reversed(rewards):
            running_return = reward + self.algorithm_config.gamma * running_return
            returns.append(running_return)
        returns.reverse()
        return returns

    def _state_batch(self, state_indices: list[int]) -> torch.Tensor:
        index_tensor = torch.tensor(state_indices, dtype=torch.long, device=self.device)
        return torch.nn.functional.one_hot(index_tensor, num_classes=self.state_dim).to(torch.float32)

    def _build_policy_grid(self) -> list[list[str]]:
        grid: list[list[str]] = []
        for row in range(self.env.size):
            current_row: list[str] = []
            for col in range(self.env.size):
                special_token = self.env.cell_token(row, col)
                if special_token is not None:
                    current_row.append(special_token)
                    continue

                state_index = row * self.env.size + col
                with torch.no_grad():
                    logits = self.policy_net(self._state_batch([state_index]))
                    action_index = int(torch.argmax(logits, dim=1).item())
                current_row.append(ACTION_CODES[action_index])
            grid.append(current_row)
        return grid
