from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol

from app.schemas.experiment import GridPosition


ACTION_DELTAS: Final[tuple[tuple[int, int], ...]] = ((-1, 0), (0, 1), (1, 0), (0, -1))
ACTION_CODES: Final[tuple[str, ...]] = ("U", "R", "D", "L")


@dataclass(slots=True)
class TransitionResult:
    state: GridPosition
    reward: float
    done: bool
    terminated_by: str


class DiscreteGridEnvironment(Protocol):
    environment_id: str
    rows: int
    cols: int
    state_count: int

    def reset(self) -> GridPosition:
        ...

    def state_to_index(self, state: GridPosition) -> int:
        ...

    def step(self, action: int) -> TransitionResult:
        ...

    def cell_token(self, row: int, col: int) -> str | None:
        ...
