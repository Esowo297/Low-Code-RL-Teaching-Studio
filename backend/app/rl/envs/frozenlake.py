from __future__ import annotations

import numpy as np

from app.rl.envs.base import ACTION_DELTAS, TransitionResult
from app.schemas.experiment import FrozenLakeConfig, GridPosition


class FrozenLakeEnv:
    def __init__(self, config: FrozenLakeConfig, *, seed: int | None = None) -> None:
        self.config = config
        self.environment_id = config.environment_id
        self.rows = config.rows
        self.cols = config.cols
        self.state_count = config.rows * config.cols
        self.current_state = config.start.model_copy(deep=True)
        self.hole_cells = {(cell.row, cell.col) for cell in config.holes}
        self.goal_cell = (config.goal.row, config.goal.col)
        self.start_cell = (config.start.row, config.start.col)
        self.rng = np.random.default_rng(seed)

    def reset(self) -> GridPosition:
        self.current_state = self.config.start.model_copy(deep=True)
        return self.current_state.model_copy(deep=True)

    def state_to_index(self, state: GridPosition) -> int:
        return state.row * self.cols + state.col

    def step(self, action: int) -> TransitionResult:
        sampled_action = self._sample_action(action)
        row_delta, col_delta = ACTION_DELTAS[sampled_action]
        candidate_row = self.current_state.row + row_delta
        candidate_col = self.current_state.col + col_delta

        reward = self.config.rewards.step_penalty
        done = False
        terminated_by = "step"

        out_of_bounds = not (0 <= candidate_row < self.rows and 0 <= candidate_col < self.cols)
        if out_of_bounds:
            candidate = self.current_state.model_copy(deep=True)
            reward = self.config.rewards.wall_penalty
            terminated_by = "blocked"
        else:
            candidate = GridPosition(row=candidate_row, col=candidate_col)
            candidate_cell = (candidate.row, candidate.col)

            if candidate_cell == self.goal_cell:
                reward = self.config.rewards.goal_reward
                done = True
                terminated_by = "goal"
            elif candidate_cell in self.hole_cells:
                reward = self.config.rewards.hole_penalty
                done = True
                terminated_by = "hole"

        self.current_state = candidate
        return TransitionResult(
            state=candidate.model_copy(deep=True),
            reward=reward,
            done=done,
            terminated_by=terminated_by,
        )

    def cell_token(self, row: int, col: int) -> str | None:
        cell = (row, col)
        if cell == self.start_cell:
            return "START"
        if cell == self.goal_cell:
            return "GOAL"
        if cell in self.hole_cells:
            return "HOLE"
        return None

    def _sample_action(self, action: int) -> int:
        if self.config.slip_probability <= 0.0:
            return action
        if self.rng.random() >= self.config.slip_probability:
            return action
        turn = -1 if self.rng.random() < 0.5 else 1
        return (action + turn) % len(ACTION_DELTAS)
