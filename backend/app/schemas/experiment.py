from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


def default_obstacles() -> list["GridPosition"]:
    return [
        GridPosition(row=1, col=2),
        GridPosition(row=2, col=2),
        GridPosition(row=4, col=2),
        GridPosition(row=4, col=3),
    ]


def default_traps() -> list["GridPosition"]:
    return [
        GridPosition(row=2, col=4),
        GridPosition(row=4, col=1),
    ]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


EnvironmentId: TypeAlias = Literal["gridworld", "cliffwalking", "windygridworld", "frozenlake"]


class GridPosition(StrictModel):
    row: int = Field(ge=0, le=19)
    col: int = Field(ge=0, le=19)


class RewardConfig(StrictModel):
    step_penalty: float = -1.0
    goal_reward: float = 20.0
    wall_penalty: float = -3.0
    trap_penalty: float = -10.0


class CliffRewardConfig(StrictModel):
    step_penalty: float = -1.0
    goal_reward: float = 0.0
    wall_penalty: float = -1.0
    cliff_penalty: float = -100.0


class WindyRewardConfig(StrictModel):
    step_penalty: float = -1.0
    goal_reward: float = 0.0
    wall_penalty: float = -1.0


class FrozenLakeRewardConfig(StrictModel):
    step_penalty: float = 0.0
    goal_reward: float = 1.0
    wall_penalty: float = 0.0
    hole_penalty: float = 0.0


class EnvironmentConfigBase(StrictModel):
    environment_id: EnvironmentId


class GridWorldConfig(EnvironmentConfigBase):
    environment_id: Literal["gridworld"] = "gridworld"
    rows: int = Field(default=6, ge=4, le=12)
    cols: int = Field(default=6, ge=4, le=12)
    start: GridPosition = Field(default_factory=lambda: GridPosition(row=0, col=0))
    goal: GridPosition = Field(default_factory=lambda: GridPosition(row=5, col=5))
    obstacles: list[GridPosition] = Field(default_factory=default_obstacles)
    traps: list[GridPosition] = Field(default_factory=default_traps)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_size(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        size = payload.pop("size", None)
        if size is not None:
            payload.setdefault("rows", size)
            payload.setdefault("cols", size)

        payload.setdefault("environment_id", "gridworld")

        rows = payload.get("rows", 6)
        cols = payload.get("cols", 6)

        if "goal" not in payload:
            payload["goal"] = {"row": rows - 1, "col": cols - 1}
        if "obstacles" not in payload:
            payload["obstacles"] = default_obstacles() if rows == 6 and cols == 6 else []
        if "traps" not in payload:
            payload["traps"] = default_traps() if rows == 6 and cols == 6 else []

        return payload

    @model_validator(mode="after")
    def validate_cells(self) -> "GridWorldConfig":
        if self.rows != self.cols:
            raise ValueError("gridworld currently requires rows and cols to be equal")

        reserved: dict[tuple[int, int], str] = {}

        for label, cell in [("start", self.start), ("goal", self.goal)]:
            self._assert_in_bounds(label, cell)
            reserved[(cell.row, cell.col)] = label

        for label, collection in [("obstacle", self.obstacles), ("trap", self.traps)]:
            for cell in collection:
                self._assert_in_bounds(label, cell)
                key = (cell.row, cell.col)
                if key in reserved:
                    raise ValueError(f"{label} cell {key} overlaps with {reserved[key]}")
                reserved[key] = label

        return self

    @computed_field(return_type=int)
    @property
    def size(self) -> int:
        return self.rows

    def _assert_in_bounds(self, label: str, cell: GridPosition) -> None:
        if cell.row >= self.rows or cell.col >= self.cols:
            raise ValueError(f"{label} cell ({cell.row}, {cell.col}) is outside a {self.rows}x{self.cols} grid")


def default_cliff_cells_for_shape(rows: int, cols: int) -> list[GridPosition]:
    if rows < 1 or cols < 3:
        return []
    cliff_row = rows - 1
    return [GridPosition(row=cliff_row, col=col) for col in range(1, cols - 1)]


class CliffWalkingConfig(EnvironmentConfigBase):
    environment_id: Literal["cliffwalking"] = "cliffwalking"
    rows: int = Field(default=4, ge=4, le=12)
    cols: int = Field(default=12, ge=4, le=20)
    start: GridPosition = Field(default_factory=lambda: GridPosition(row=3, col=0))
    goal: GridPosition = Field(default_factory=lambda: GridPosition(row=3, col=11))
    cliffs: list[GridPosition] = Field(default_factory=lambda: default_cliff_cells_for_shape(4, 12))
    rewards: CliffRewardConfig = Field(default_factory=CliffRewardConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload.setdefault("environment_id", "cliffwalking")

        rows = payload.get("rows", 4)
        cols = payload.get("cols", 12)

        if "start" not in payload:
            payload["start"] = {"row": rows - 1, "col": 0}
        if "goal" not in payload:
            payload["goal"] = {"row": rows - 1, "col": cols - 1}
        if "cliffs" not in payload:
            payload["cliffs"] = default_cliff_cells_for_shape(rows, cols)

        return payload

    @model_validator(mode="after")
    def validate_cells(self) -> "CliffWalkingConfig":
        reserved: dict[tuple[int, int], str] = {}

        for label, cell in [("start", self.start), ("goal", self.goal)]:
            self._assert_in_bounds(label, cell)
            reserved[(cell.row, cell.col)] = label

        for cell in self.cliffs:
            self._assert_in_bounds("cliff", cell)
            key = (cell.row, cell.col)
            if key in reserved:
                raise ValueError(f"cliff cell {key} overlaps with {reserved[key]}")
            reserved[key] = "cliff"

        return self

    def _assert_in_bounds(self, label: str, cell: GridPosition) -> None:
        if cell.row >= self.rows or cell.col >= self.cols:
            raise ValueError(f"{label} cell ({cell.row}, {cell.col}) is outside a {self.rows}x{self.cols} grid")


def default_windy_start(rows: int) -> GridPosition:
    return GridPosition(row=min(3, rows - 1), col=0)


def default_windy_goal(rows: int, cols: int) -> GridPosition:
    return GridPosition(row=min(3, rows - 1), col=min(7, cols - 1))


def default_wind_strengths_for_cols(cols: int) -> list[int]:
    base = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
    if cols <= len(base):
        return base[:cols]
    return [*base, *([0] * (cols - len(base)))]


class WindyGridWorldConfig(EnvironmentConfigBase):
    environment_id: Literal["windygridworld"] = "windygridworld"
    rows: int = Field(default=7, ge=4, le=12)
    cols: int = Field(default=10, ge=5, le=20)
    start: GridPosition = Field(default_factory=lambda: default_windy_start(7))
    goal: GridPosition = Field(default_factory=lambda: default_windy_goal(7, 10))
    wind_strengths: list[int] = Field(default_factory=lambda: default_wind_strengths_for_cols(10))
    rewards: WindyRewardConfig = Field(default_factory=WindyRewardConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload.setdefault("environment_id", "windygridworld")

        rows = payload.get("rows", 7)
        cols = payload.get("cols", 10)

        if "start" not in payload:
            start = default_windy_start(rows)
            payload["start"] = {"row": start.row, "col": start.col}
        if "goal" not in payload:
            goal = default_windy_goal(rows, cols)
            payload["goal"] = {"row": goal.row, "col": goal.col}
        if "wind_strengths" not in payload:
            payload["wind_strengths"] = default_wind_strengths_for_cols(cols)

        return payload

    @model_validator(mode="after")
    def validate_shape(self) -> "WindyGridWorldConfig":
        self._assert_in_bounds("start", self.start)
        self._assert_in_bounds("goal", self.goal)
        if (self.start.row, self.start.col) == (self.goal.row, self.goal.col):
            raise ValueError("start and goal cannot be the same cell")
        if len(self.wind_strengths) != self.cols:
            raise ValueError(f"wind_strengths length {len(self.wind_strengths)} does not match cols={self.cols}")
        for index, strength in enumerate(self.wind_strengths):
            if strength < 0:
                raise ValueError(f"wind_strengths[{index}] must be greater than or equal to 0")
            if strength >= self.rows:
                raise ValueError(f"wind_strengths[{index}]={strength} is too large for rows={self.rows}")
        return self

    def _assert_in_bounds(self, label: str, cell: GridPosition) -> None:
        if cell.row >= self.rows or cell.col >= self.cols:
            raise ValueError(f"{label} cell ({cell.row}, {cell.col}) is outside a {self.rows}x{self.cols} grid")


def default_frozen_start() -> GridPosition:
    return GridPosition(row=0, col=0)


def default_frozen_goal(rows: int, cols: int) -> GridPosition:
    return GridPosition(row=rows - 1, col=cols - 1)


def default_holes_for_shape(rows: int, cols: int) -> list[GridPosition]:
    if rows == 4 and cols == 4:
        return [
            GridPosition(row=1, col=1),
            GridPosition(row=1, col=3),
            GridPosition(row=2, col=3),
            GridPosition(row=3, col=0),
        ]
    return []


class FrozenLakeConfig(EnvironmentConfigBase):
    environment_id: Literal["frozenlake"] = "frozenlake"
    rows: int = Field(default=4, ge=4, le=12)
    cols: int = Field(default=4, ge=4, le=12)
    start: GridPosition = Field(default_factory=default_frozen_start)
    goal: GridPosition = Field(default_factory=lambda: default_frozen_goal(4, 4))
    holes: list[GridPosition] = Field(default_factory=lambda: default_holes_for_shape(4, 4))
    slip_probability: float = Field(default=0.2, ge=0.0, le=1.0)
    rewards: FrozenLakeRewardConfig = Field(default_factory=FrozenLakeRewardConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_defaults(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        payload.setdefault("environment_id", "frozenlake")

        rows = payload.get("rows", 4)
        cols = payload.get("cols", 4)

        if "start" not in payload:
            start = default_frozen_start()
            payload["start"] = {"row": start.row, "col": start.col}
        if "goal" not in payload:
            goal = default_frozen_goal(rows, cols)
            payload["goal"] = {"row": goal.row, "col": goal.col}
        if "holes" not in payload:
            payload["holes"] = default_holes_for_shape(rows, cols)

        return payload

    @model_validator(mode="after")
    def validate_cells(self) -> "FrozenLakeConfig":
        reserved: dict[tuple[int, int], str] = {}

        for label, cell in [("start", self.start), ("goal", self.goal)]:
            self._assert_in_bounds(label, cell)
            key = (cell.row, cell.col)
            if key in reserved:
                raise ValueError(f"{label} cell {key} overlaps with {reserved[key]}")
            reserved[key] = label

        for cell in self.holes:
            self._assert_in_bounds("hole", cell)
            key = (cell.row, cell.col)
            if key in reserved:
                raise ValueError(f"hole cell {key} overlaps with {reserved[key]}")
            reserved[key] = "hole"

        return self

    def _assert_in_bounds(self, label: str, cell: GridPosition) -> None:
        if cell.row >= self.rows or cell.col >= self.cols:
            raise ValueError(f"{label} cell ({cell.row}, {cell.col}) is outside a {self.rows}x{self.cols} grid")


class QLearningConfig(StrictModel):
    learning_rate: float = Field(default=0.2, gt=0.0, le=1.0)
    gamma: float = Field(default=0.92, gt=0.0, le=0.999)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_min: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.98, gt=0.0, le=1.0)
    max_steps_per_episode: int = Field(default=80, ge=10, le=400)

    @model_validator(mode="after")
    def validate_schedule(self) -> "QLearningConfig":
        if self.epsilon_min > self.epsilon_start:
            raise ValueError("epsilon_min must be less than or equal to epsilon_start")
        return self


class SARSAConfig(QLearningConfig):
    pass


class DQNConfig(StrictModel):
    learning_rate: float = Field(default=0.001, gt=0.0, le=0.1)
    gamma: float = Field(default=0.95, gt=0.0, le=0.999)
    epsilon_start: float = Field(default=1.0, ge=0.0, le=1.0)
    epsilon_min: float = Field(default=0.05, ge=0.0, le=1.0)
    epsilon_decay: float = Field(default=0.992, gt=0.0, le=1.0)
    max_steps_per_episode: int = Field(default=80, ge=10, le=400)
    batch_size: int = Field(default=32, ge=8, le=256)
    replay_buffer_size: int = Field(default=2000, ge=200, le=50000)
    target_sync_interval: int = Field(default=40, ge=1, le=1000)
    warmup_steps: int = Field(default=80, ge=1, le=5000)
    hidden_dim: int = Field(default=64, ge=16, le=512)

    @model_validator(mode="after")
    def validate_schedule(self) -> "DQNConfig":
        if self.epsilon_min > self.epsilon_start:
            raise ValueError("epsilon_min must be less than or equal to epsilon_start")
        if self.batch_size > self.replay_buffer_size:
            raise ValueError("batch_size must be less than or equal to replay_buffer_size")
        if self.warmup_steps > self.replay_buffer_size:
            raise ValueError("warmup_steps must be less than or equal to replay_buffer_size")
        return self


class ReinforceConfig(StrictModel):
    learning_rate: float = Field(default=0.01, gt=0.0, le=0.1)
    gamma: float = Field(default=0.95, gt=0.0, le=0.999)
    max_steps_per_episode: int = Field(default=80, ge=10, le=400)
    hidden_dim: int = Field(default=64, ge=16, le=512)


class TrainingConfig(StrictModel):
    episodes: int = Field(default=120, ge=20, le=2000)
    seed: int = Field(default=7, ge=0, le=99999)
    trace_frequency: int = Field(default=30, ge=1, le=500)


SubmissionRole: TypeAlias = Literal["teacher", "student"]
EnvironmentConfig: TypeAlias = Annotated[
    GridWorldConfig | CliffWalkingConfig | WindyGridWorldConfig | FrozenLakeConfig,
    Field(discriminator="environment_id"),
]


class ExperimentRequestBase(StrictModel):
    name: str = Field(default="Reinforcement Learning Experiment", min_length=3, max_length=80)
    submitted_by: str = Field(default="Anonymous Student", min_length=2, max_length=40)
    submission_role: SubmissionRole = "student"
    assignment_id: str | None = Field(default=None, max_length=60)
    assignment_title: str | None = Field(default=None, max_length=120)
    environment_id: EnvironmentId = "gridworld"
    persist_result: bool = True
    env_config: EnvironmentConfig = Field(default_factory=GridWorldConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_env_config_discriminator(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        env_config = payload.get("env_config")
        environment_id = payload.get("environment_id", "gridworld")
        if isinstance(env_config, dict) and "environment_id" not in env_config:
            nested_config = dict(env_config)
            nested_config["environment_id"] = environment_id
            payload["env_config"] = nested_config
        return payload

    @model_validator(mode="after")
    def validate_environment_binding(self) -> "ExperimentRequestBase":
        if self.environment_id != self.env_config.environment_id:
            raise ValueError(
                f"environment_id '{self.environment_id}' does not match env_config '{self.env_config.environment_id}'"
            )
        return self


class QLearningExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="Q-Learning Demo", min_length=3, max_length=80)
    algorithm_id: Literal["q_learning"] = "q_learning"
    algorithm_config: QLearningConfig = Field(default_factory=QLearningConfig)


class SARSAExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="SARSA Demo", min_length=3, max_length=80)
    algorithm_id: Literal["sarsa"] = "sarsa"
    algorithm_config: SARSAConfig = Field(default_factory=SARSAConfig)


class DQNExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="DQN Demo", min_length=3, max_length=80)
    algorithm_id: Literal["dqn"] = "dqn"
    algorithm_config: DQNConfig = Field(default_factory=DQNConfig)


class ReinforceExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="REINFORCE Demo", min_length=3, max_length=80)
    algorithm_id: Literal["reinforce"] = "reinforce"
    algorithm_config: ReinforceConfig = Field(default_factory=ReinforceConfig)


ExperimentRequestType: TypeAlias = (
    QLearningExperimentRequest
    | SARSAExperimentRequest
    | DQNExperimentRequest
    | ReinforceExperimentRequest
)
ExperimentRequest: TypeAlias = Annotated[ExperimentRequestType, Field(discriminator="algorithm_id")]


class EpisodeMetric(StrictModel):
    episode: int
    reward: float
    td_error: float
    epsilon: float
    steps: int
    success: bool


class PathTrace(StrictModel):
    episode: int
    success: bool
    total_reward: float
    path: list[GridPosition]


class ExperimentSummary(StrictModel):
    average_reward: float
    best_reward: float
    success_rate: float
    stable_success_rate: float


class GridEnvironmentView(StrictModel):
    view_type: Literal["grid"] = "grid"
    rows: int = Field(ge=1, le=40)
    cols: int = Field(ge=1, le=40)
    cells: list[list[str]]

    @model_validator(mode="after")
    def validate_shape(self) -> "GridEnvironmentView":
        if len(self.cells) != self.rows:
            raise ValueError(f"environment_view row count {len(self.cells)} does not match rows={self.rows}")
        for row in self.cells:
            if len(row) != self.cols:
                raise ValueError(f"environment_view column count {len(row)} does not match cols={self.cols}")
        return self


EnvironmentView: TypeAlias = GridEnvironmentView


class ExperimentResult(StrictModel):
    run_id: str
    created_at: str
    status: str = "completed"
    request: ExperimentRequestType = Field(discriminator="algorithm_id")
    summary: ExperimentSummary
    metrics: list[EpisodeMetric]
    path_traces: list[PathTrace]
    environment_view: EnvironmentView | None = None
    policy_grid: list[list[str]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_rendering(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        payload = dict(data)
        policy_grid = payload.get("policy_grid")
        environment_view = payload.get("environment_view")

        if environment_view is None and policy_grid:
            rows = len(policy_grid)
            cols = max((len(row) for row in policy_grid), default=0)
            payload["environment_view"] = {
                "view_type": "grid",
                "rows": rows,
                "cols": cols,
                "cells": policy_grid,
            }
            return payload

        if "policy_grid" not in payload and environment_view is not None:
            if isinstance(environment_view, GridEnvironmentView):
                if environment_view.view_type == "grid":
                    payload["policy_grid"] = environment_view.cells
            elif isinstance(environment_view, dict) and environment_view.get("view_type") == "grid":
                payload["policy_grid"] = environment_view.get("cells", [])

        return payload


class ExperimentHistoryEntry(StrictModel):
    run_id: str
    name: str
    submitted_by: str
    submission_role: SubmissionRole
    assignment_id: str | None = None
    assignment_title: str | None = None
    created_at: str
    environment_id: str
    algorithm_id: str
    average_reward: float
    success_rate: float


class ParameterDefinition(StrictModel):
    key: str
    label: str
    default: Any
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    help_text: str


class ModuleDefinition(StrictModel):
    id: str
    name: str
    description: str
    parameters: list[ParameterDefinition]


class CatalogResponse(StrictModel):
    environments: list[ModuleDefinition]
    algorithms: list[ModuleDefinition]


class BenchmarkThreshold(StrictModel):
    metric_id: Literal["average_reward", "best_reward", "success_rate", "stable_success_rate"]
    label: str
    min_value: float
    help_text: str


class BenchmarkDraft(StrictModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=240)
    teacher_note: str = Field(min_length=1, max_length=240)
    request: ExperimentRequestType = Field(discriminator="algorithm_id")
    thresholds: list[BenchmarkThreshold] = Field(min_length=1, max_length=4)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "BenchmarkDraft":
        metric_ids = [threshold.metric_id for threshold in self.thresholds]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("benchmark thresholds contain duplicate metric ids")
        return self


class BenchmarkPreset(StrictModel):
    id: str
    name: str
    description: str
    teacher_note: str
    request: ExperimentRequestType = Field(discriminator="algorithm_id")
    thresholds: list[BenchmarkThreshold]
    is_builtin: bool = True
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode="after")
    def validate_thresholds(self) -> "BenchmarkPreset":
        metric_ids = [threshold.metric_id for threshold in self.thresholds]
        if len(metric_ids) != len(set(metric_ids)):
            raise ValueError("benchmark thresholds contain duplicate metric ids")
        return self


class BenchmarkCatalogResponse(StrictModel):
    benchmarks: list[BenchmarkPreset]


class AssignmentPreset(StrictModel):
    id: str
    title: str
    summary: str
    instructions: str
    learning_goals: list[str]
    benchmark_id: str | None = None
    request: ExperimentRequestType = Field(discriminator="algorithm_id")


class AssignmentCatalogResponse(StrictModel):
    assignments: list[AssignmentPreset]


class ExperimentReportRequest(StrictModel):
    result: ExperimentResult
    benchmark_id: str | None = None


class AlgorithmAnalyticsEntry(StrictModel):
    algorithm_id: str
    run_count: int
    average_reward: float
    average_success_rate: float


class StudentAnalyticsEntry(StrictModel):
    submitted_by: str
    run_count: int
    average_reward: float
    average_success_rate: float
    best_success_rate: float
    latest_created_at: str
    benchmark_pass_count: int | None = None


class BenchmarkAnalyticsSummary(StrictModel):
    benchmark_id: str
    benchmark_name: str
    evaluated_runs: int
    pass_count: int
    pass_rate: float


class AssignmentAnalyticsEntry(StrictModel):
    assignment_id: str
    assignment_title: str
    run_count: int
    student_runs: int
    distinct_submitters: int
    average_reward: float
    average_success_rate: float
    benchmark_id: str | None = None
    benchmark_pass_rate: float | None = None


class EnvironmentAnalyticsEntry(StrictModel):
    environment_id: str
    run_count: int
    student_runs: int
    distinct_submitters: int
    average_reward: float
    average_success_rate: float
    best_success_rate: float


class ClassroomAnalyticsResponse(StrictModel):
    total_runs: int
    student_runs: int
    teacher_runs: int
    distinct_submitters: int
    average_reward: float
    average_success_rate: float
    benchmark: BenchmarkAnalyticsSummary | None = None
    assignment_filter_id: str | None = None
    algorithms: list[AlgorithmAnalyticsEntry]
    assignments: list[AssignmentAnalyticsEntry]
    environments: list[EnvironmentAnalyticsEntry]
    students: list[StudentAnalyticsEntry]
