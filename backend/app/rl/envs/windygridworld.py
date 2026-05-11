from __future__ import annotations

from app.rl.envs.base import ACTION_DELTAS, TransitionResult
from app.schemas.experiment import GridPosition, WindyGridWorldConfig


class WindyGridWorldEnv:
    def __init__(self, config: WindyGridWorldConfig) -> None:
        self.config = config
        self.environment_id = config.environment_id
        self.rows = config.rows
        self.cols = config.cols
        self.state_count = config.rows * config.cols
        self.current_state = config.start.model_copy(deep=True)
        self.goal_cell = (config.goal.row, config.goal.col)
        self.start_cell = (config.start.row, config.start.col)

    def reset(self) -> GridPosition:
        self.current_state = self.config.start.model_copy(deep=True)
        return self.current_state.model_copy(deep=True)

    def state_to_index(self, state: GridPosition) -> int:
        return state.row * self.cols + state.col

    def step(self, action: int) -> TransitionResult:
        row_delta, col_delta = ACTION_DELTAS[action]
        raw_row = self.current_state.row + row_delta
        raw_col = self.current_state.col + col_delta

        next_row = min(max(raw_row, 0), self.rows - 1)
        next_col = min(max(raw_col, 0), self.cols - 1)
        blocked = raw_row != next_row or raw_col != next_col

        wind_strength = self.config.wind_strengths[next_col]
        winded_row = next_row - wind_strength
        if winded_row < 0:
            blocked = True
        next_row = max(winded_row, 0)

        reward = self.config.rewards.wall_penalty if blocked else self.config.rewards.step_penalty
        done = False
        terminated_by = "blocked" if blocked else "step"
        candidate = GridPosition(row=next_row, col=next_col)

        if (candidate.row, candidate.col) == self.goal_cell:
            reward = self.config.rewards.goal_reward
            done = True
            terminated_by = "goal"

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
        return None
