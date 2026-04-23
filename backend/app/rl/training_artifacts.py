from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeAlias

from app.schemas.experiment import EpisodeMetric, PathTrace


@dataclass(slots=True)
class TrainingArtifacts:
    metrics: list[EpisodeMetric]
    path_traces: list[PathTrace]
    policy_grid: list[list[str]]
    average_reward: float
    best_reward: float
    success_rate: float
    stable_success_rate: float


@dataclass(slots=True)
class ProgressUpdate:
    metric: EpisodeMetric
    latest_trace: PathTrace | None
    completed_episodes: int
    total_episodes: int
    running_average_reward: float
    running_success_rate: float


AsyncProgressCallback: TypeAlias = Callable[[ProgressUpdate], Awaitable[None]]


class TrainingCancelledError(Exception):
    """Raised when the client cancels a streamed training run."""


class StreamController:
    def __init__(self) -> None:
        self._resume_event = asyncio.Event()
        self._resume_event.set()
        self._cancelled = False
        self._state = "running"

    @property
    def state(self) -> str:
        return self._state

    def pause(self) -> None:
        if self._cancelled:
            return
        self._state = "paused"
        self._resume_event.clear()

    def resume(self) -> None:
        if self._cancelled:
            return
        self._state = "running"
        self._resume_event.set()

    def cancel(self) -> None:
        self._cancelled = True
        self._state = "cancelled"
        self._resume_event.set()

    async def checkpoint(self) -> None:
        await asyncio.sleep(0)
        if self._cancelled:
            raise TrainingCancelledError("Training cancelled by user")
        await self._resume_event.wait()
        if self._cancelled:
            raise TrainingCancelledError("Training cancelled by user")
