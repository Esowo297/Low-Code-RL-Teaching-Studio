export type AlgorithmId = 'q_learning' | 'dqn'
export type SubmissionRole = 'teacher' | 'student'

export interface GridPosition {
  row: number
  col: number
}

export interface RewardConfig {
  step_penalty: number
  goal_reward: number
  wall_penalty: number
  trap_penalty: number
}

export interface GridWorldConfig {
  size: number
  start: GridPosition
  goal: GridPosition
  obstacles: GridPosition[]
  traps: GridPosition[]
  rewards: RewardConfig
}

export interface QLearningConfig {
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_min: number
  epsilon_decay: number
  max_steps_per_episode: number
}

export interface DQNConfig {
  learning_rate: number
  gamma: number
  epsilon_start: number
  epsilon_min: number
  epsilon_decay: number
  max_steps_per_episode: number
  batch_size: number
  replay_buffer_size: number
  target_sync_interval: number
  warmup_steps: number
  hidden_dim: number
}

export interface TrainingConfig {
  episodes: number
  seed: number
  trace_frequency: number
}

export interface ExperimentRequestBase {
  name: string
  submitted_by: string
  submission_role: SubmissionRole
  assignment_id?: string | null
  assignment_title?: string | null
  environment_id: string
  persist_result: boolean
  env_config: GridWorldConfig
  training: TrainingConfig
}

export interface QLearningExperimentRequest extends ExperimentRequestBase {
  algorithm_id: 'q_learning'
  algorithm_config: QLearningConfig
}

export interface DQNExperimentRequest extends ExperimentRequestBase {
  algorithm_id: 'dqn'
  algorithm_config: DQNConfig
}

export type ExperimentRequest = QLearningExperimentRequest | DQNExperimentRequest

export interface EpisodeMetric {
  episode: number
  reward: number
  td_error: number
  epsilon: number
  steps: number
  success: boolean
}

export interface PathTrace {
  episode: number
  success: boolean
  total_reward: number
  path: GridPosition[]
}

export interface ExperimentSummary {
  average_reward: number
  best_reward: number
  success_rate: number
  stable_success_rate: number
}

export interface ExperimentResult {
  run_id: string
  created_at: string
  status: string
  request: ExperimentRequest
  summary: ExperimentSummary
  metrics: EpisodeMetric[]
  path_traces: PathTrace[]
  policy_grid: string[][]
}

export interface HistoryEntry {
  run_id: string
  name: string
  submitted_by: string
  submission_role: SubmissionRole
  assignment_id?: string | null
  assignment_title?: string | null
  created_at: string
  environment_id: string
  algorithm_id: AlgorithmId
  average_reward: number
  success_rate: number
}

export interface ParameterDefinition {
  key: string
  label: string
  default: string | number | boolean
  min_value?: number | null
  max_value?: number | null
  step?: number | null
  help_text: string
}

export interface ModuleDefinition {
  id: string
  name: string
  description: string
  parameters: ParameterDefinition[]
}

export interface CatalogResponse {
  environments: ModuleDefinition[]
  algorithms: ModuleDefinition[]
}

export type BenchmarkMetricId = 'average_reward' | 'best_reward' | 'success_rate' | 'stable_success_rate'

export interface BenchmarkThreshold {
  metric_id: BenchmarkMetricId
  label: string
  min_value: number
  help_text: string
}

export interface BenchmarkPreset {
  id: string
  name: string
  description: string
  teacher_note: string
  request: ExperimentRequest
  thresholds: BenchmarkThreshold[]
}

export interface BenchmarkCatalogResponse {
  benchmarks: BenchmarkPreset[]
}

export interface AssignmentPreset {
  id: string
  title: string
  summary: string
  instructions: string
  learning_goals: string[]
  benchmark_id?: string | null
  request: ExperimentRequest
}

export interface AssignmentCatalogResponse {
  assignments: AssignmentPreset[]
}

export interface ExperimentReportRequest {
  result: ExperimentResult
  benchmark_id?: string | null
}

export interface AlgorithmAnalyticsEntry {
  algorithm_id: AlgorithmId
  run_count: number
  average_reward: number
  average_success_rate: number
}

export interface StudentAnalyticsEntry {
  submitted_by: string
  run_count: number
  average_reward: number
  average_success_rate: number
  best_success_rate: number
  latest_created_at: string
  benchmark_pass_count?: number | null
}

export interface BenchmarkAnalyticsSummary {
  benchmark_id: string
  benchmark_name: string
  evaluated_runs: number
  pass_count: number
  pass_rate: number
}

export interface AssignmentAnalyticsEntry {
  assignment_id: string
  assignment_title: string
  run_count: number
  student_runs: number
  distinct_submitters: number
  average_reward: number
  average_success_rate: number
  benchmark_id?: string | null
  benchmark_pass_rate?: number | null
}

export interface ClassroomAnalyticsResponse {
  total_runs: number
  student_runs: number
  teacher_runs: number
  distinct_submitters: number
  average_reward: number
  average_success_rate: number
  benchmark?: BenchmarkAnalyticsSummary | null
  assignment_filter_id?: string | null
  algorithms: AlgorithmAnalyticsEntry[]
  assignments: AssignmentAnalyticsEntry[]
  students: StudentAnalyticsEntry[]
}

export interface ChartPoint {
  x: number
  y: number
}

export interface ComparisonSeries {
  runId: string
  label: string
  color: string
  points: ChartPoint[]
}

export interface ExperimentStartedEvent {
  type: 'started'
  run_id: string
  created_at: string
  request: ExperimentRequest
}

export interface ExperimentMetricEvent {
  type: 'metric'
  metric: EpisodeMetric
  latest_trace: PathTrace | null
  completed_episodes: number
  total_episodes: number
  progress: number
  running_average_reward: number
  running_success_rate: number
}

export interface ExperimentControlEvent {
  type: 'control'
  state: 'running' | 'paused' | 'cancelling'
}

export interface ExperimentCompletedEvent {
  type: 'completed'
  result: ExperimentResult
}

export interface ExperimentCancelledEvent {
  type: 'cancelled'
  run_id: string
  completed_episodes: number
  total_episodes: number
  message: string
}

export interface ExperimentErrorEvent {
  type: 'error'
  message: string | Array<Record<string, unknown>>
}

export type ExperimentStreamEvent =
  | ExperimentStartedEvent
  | ExperimentMetricEvent
  | ExperimentControlEvent
  | ExperimentCompletedEvent
  | ExperimentCancelledEvent
  | ExperimentErrorEvent
