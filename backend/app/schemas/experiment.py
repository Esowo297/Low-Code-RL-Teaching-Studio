from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class GridPosition(StrictModel):
    row: int = Field(ge=0, le=19)
    col: int = Field(ge=0, le=19)


class RewardConfig(StrictModel):
    step_penalty: float = -1.0
    goal_reward: float = 20.0
    wall_penalty: float = -3.0
    trap_penalty: float = -10.0


class GridWorldConfig(StrictModel):
    size: int = Field(default=6, ge=4, le=12)
    start: GridPosition = Field(default_factory=lambda: GridPosition(row=0, col=0))
    goal: GridPosition = Field(default_factory=lambda: GridPosition(row=5, col=5))
    obstacles: list[GridPosition] = Field(default_factory=default_obstacles)
    traps: list[GridPosition] = Field(default_factory=default_traps)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="after")
    def validate_cells(self) -> "GridWorldConfig":
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

    def _assert_in_bounds(self, label: str, cell: GridPosition) -> None:
        if cell.row >= self.size or cell.col >= self.size:
            raise ValueError(f"{label} cell ({cell.row}, {cell.col}) is outside a {self.size}x{self.size} grid")


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


class ExperimentRequestBase(StrictModel):
    name: str = Field(default="GridWorld Experiment", min_length=3, max_length=80)
    submitted_by: str = Field(default="Anonymous Student", min_length=2, max_length=40)
    submission_role: SubmissionRole = "student"
    assignment_id: str | None = Field(default=None, max_length=60)
    assignment_title: str | None = Field(default=None, max_length=120)
    environment_id: Literal["gridworld"] = "gridworld"
    persist_result: bool = True
    env_config: GridWorldConfig = Field(default_factory=GridWorldConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


class QLearningExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="GridWorld Q-Learning Demo", min_length=3, max_length=80)
    algorithm_id: Literal["q_learning"] = "q_learning"
    algorithm_config: QLearningConfig = Field(default_factory=QLearningConfig)


class SARSAExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="GridWorld SARSA Demo", min_length=3, max_length=80)
    algorithm_id: Literal["sarsa"] = "sarsa"
    algorithm_config: SARSAConfig = Field(default_factory=SARSAConfig)


class DQNExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="GridWorld DQN Demo", min_length=3, max_length=80)
    algorithm_id: Literal["dqn"] = "dqn"
    algorithm_config: DQNConfig = Field(default_factory=DQNConfig)


class ReinforceExperimentRequest(ExperimentRequestBase):
    name: str = Field(default="GridWorld REINFORCE Demo", min_length=3, max_length=80)
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


class ExperimentResult(StrictModel):
    run_id: str
    created_at: str
    status: str = "completed"
    request: ExperimentRequestType = Field(discriminator="algorithm_id")
    summary: ExperimentSummary
    metrics: list[EpisodeMetric]
    path_traces: list[PathTrace]
    policy_grid: list[list[str]]


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


class BenchmarkPreset(StrictModel):
    id: str
    name: str
    description: str
    teacher_note: str
    request: ExperimentRequestType = Field(discriminator="algorithm_id")
    thresholds: list[BenchmarkThreshold]


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
    students: list[StudentAnalyticsEntry]
