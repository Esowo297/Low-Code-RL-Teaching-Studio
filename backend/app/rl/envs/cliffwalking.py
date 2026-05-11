from __future__ import annotations

from app.rl.envs.base import ACTION_DELTAS, TransitionResult
from app.schemas.experiment import CliffWalkingConfig, GridPosition


class CliffWalkingEnv:
    def __init__(self, config: CliffWalkingConfig) -> None:
        self.config = config
        self.environment_id = config.environment_id
        self.rows = config.rows
        self.cols = config.cols
        self.state_count = config.rows * config.cols
        self.current_state = config.start.model_copy(deep=True)
        self.cliff_cells = {(cell.row, cell.col) for cell in config.cliffs}
        self.goal_cell = (config.goal.row, config.goal.col)
        self.start_cell = (config.start.row, config.start.col)

    def reset(self) -> GridPosition:
        self.current_state = self.config.start.model_copy(deep=True)
        return self.current_state.model_copy(deep=True)

    def state_to_index(self, state: GridPosition) -> int:
        return state.row * self.cols + state.col

    def step(self, action: int) -> TransitionResult:
        row_delta, col_delta = ACTION_DELTAS[action]
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
            elif candidate_cell in self.cliff_cells:
                reward = self.config.rewards.cliff_penalty
                candidate = self.config.start.model_copy(deep=True)
                terminated_by = "cliff"

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
        if cell in self.cliff_cells:
            return "CLIFF"
        return None
