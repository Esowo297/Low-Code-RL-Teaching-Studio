from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.schemas.experiment import GridPosition, GridWorldConfig


ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_CODES: Final[tuple[str, ...]] = ("U", "R", "D", "L")


@dataclass(slots=True)
class TransitionResult:
    state: GridPosition
    reward: float
    done: bool
    terminated_by: str


class GridWorldEnv:
    def __init__(self, config: GridWorldConfig) -> None:
        self.config = config
        self.size = config.size
        self.current_state = config.start.model_copy(deep=True)
        self.obstacle_cells = {(cell.row, cell.col) for cell in config.obstacles}
        self.trap_cells = {(cell.row, cell.col) for cell in config.traps}
        self.goal_cell = (config.goal.row, config.goal.col)
        self.start_cell = (config.start.row, config.start.col)

    def reset(self) -> GridPosition:
        self.current_state = self.config.start.model_copy(deep=True)
        return self.current_state.model_copy(deep=True)

    def state_to_index(self, state: GridPosition) -> int:
        return state.row * self.size + state.col

    def step(self, action: int) -> TransitionResult:
        row_delta, col_delta = ACTION_DELTAS[action]
        candidate_row = self.current_state.row + row_delta
        candidate_col = self.current_state.col + col_delta

        reward = self.config.rewards.step_penalty
        done = False
        terminated_by = "step"

        out_of_bounds = not (0 <= candidate_row < self.size and 0 <= candidate_col < self.size)
        blocked = (candidate_row, candidate_col) in self.obstacle_cells

        if out_of_bounds or blocked:
            candidate = self.current_state.model_copy(deep=True)
            reward = self.config.rewards.wall_penalty
            terminated_by = "blocked"
        else:
            candidate = GridPosition(row=candidate_row, col=candidate_col)

            if (candidate.row, candidate.col) == self.goal_cell:
                reward = self.config.rewards.goal_reward
                done = True
                terminated_by = "goal"
            elif (candidate.row, candidate.col) in self.trap_cells:
                reward = self.config.rewards.trap_penalty
                done = True
                terminated_by = "trap"

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
        if cell in self.obstacle_cells:
            return "BLOCK"
        if cell in self.trap_cells:
            return "TRAP"
        return None
