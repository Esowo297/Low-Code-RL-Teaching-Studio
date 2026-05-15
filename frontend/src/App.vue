<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

import type { ExperimentStreamSession } from './api'
import {
  createBenchmark,
  deleteBenchmark,
  getAssignments,
  getBenchmarks,
  getCatalog,
  getClassroomAnalytics,
  getExperiment,
  getHistory,
  renderExperimentReport,
  runExperimentStream,
  updateBenchmark,
} from './api'
import ComparisonChart from './components/ComparisonChart.vue'
import MetricChart from './components/MetricChart.vue'
import PolicyGrid from './components/PolicyGrid.vue'
import TraceReplayGrid from './components/TraceReplayGrid.vue'
import type {
  AlgorithmId,
  AssignmentCatalogResponse,
  AssignmentPreset,
  BenchmarkDraft,
  ClassroomAnalyticsResponse,
  BenchmarkCatalogResponse,
  BenchmarkPreset,
  BenchmarkThreshold,
  CatalogResponse,
  CliffRewardConfig,
  CliffWalkingConfig,
  ComparisonSeries,
  DQNConfig,
  DQNExperimentRequest,
  EnvironmentConfig,
  EnvironmentId,
  EpisodeMetric,
  ExperimentControlEvent,
  ExperimentRequest,
  ExperimentResult,
  ExperimentStartedEvent,
  FrozenLakeConfig,
  FrozenLakeRewardConfig,
  GridPosition,
  GridWorldConfig,
  HistoryEntry,
  PathTrace,
  QLearningConfig,
  QLearningExperimentRequest,
  ReinforceConfig,
  ReinforceExperimentRequest,
  SARSAConfig,
  SARSAExperimentRequest,
  SubmissionRole,
  WindyGridWorldConfig,
  WindyRewardConfig,
} from './types'

interface BenchmarkCheckRow {
  label: string
  passed: boolean
  expected: string
  actual: string
  helpText: string
}

interface BenchmarkEvaluation {
  passed: boolean
  compatibilityChecks: BenchmarkCheckRow[]
  thresholdChecks: BenchmarkCheckRow[]
}

interface BenchmarkOverview {
  passed: boolean
  headline: string
  message: string
}

interface BenchmarkOverviewStat {
  label: string
  value: string
  tone: 'pass' | 'fail' | 'neutral'
}

interface BenchmarkMetricCard {
  label: string
  passed: boolean
  actual: string
  expected: string
  deltaText: string
  helpText: string
}

const catalog = ref<CatalogResponse | null>(null)
const assignmentCatalog = ref<AssignmentCatalogResponse | null>(null)
const benchmarkCatalog = ref<BenchmarkCatalogResponse | null>(null)
const analytics = ref<ClassroomAnalyticsResponse | null>(null)
const history = ref<HistoryEntry[]>([])
const currentResult = ref<ExperimentResult | null>(null)
const loading = ref(false)
const analyticsLoading = ref(false)
const errorMessage = ref('')
const streamNotice = ref('')
const selectedAlgorithm = ref<AlgorithmId>('q_learning')
const selectedEnvironment = ref<EnvironmentId>('gridworld')
const streamState = ref<'idle' | 'running' | 'paused' | 'cancelling'>('idle')
const liveRunId = ref('')
const liveMetrics = ref<EpisodeMetric[]>([])
const liveLatestTrace = ref<PathTrace | null>(null)
const liveProgress = ref(0)
const liveCompletedEpisodes = ref(0)
const liveTotalEpisodes = ref(0)
const liveAverageReward = ref(0)
const liveSuccessRate = ref(0)
const selectedComparisonIds = ref<string[]>([])
const comparisonResults = ref<Record<string, ExperimentResult>>({})
const comparisonLoadingIds = ref<string[]>([])
const selectedAssignmentId = ref('')
const selectedBenchmarkId = ref('')
const viewerRole = ref<SubmissionRole>('student')
const submittedBy = ref('学生A')
const reportExporting = ref(false)
const benchmarkSaving = ref(false)
const benchmarkDeleting = ref(false)
const replaySourceRunId = ref('')
const replayTraceEpisode = ref(0)
const replayStepIndex = ref(0)
const replayPlaying = ref(false)
let streamSession: ExperimentStreamSession | null = null
let replayTimer: number | null = null

const obstacleText = ref('1:2,2:2,4:2,4:3')
const trapText = ref('2:4,4:1')
const cliffText = ref('3:1,3:2,3:3,3:4,3:5,3:6,3:7,3:8,3:9,3:10')
const windStrengthText = ref('0,0,0,1,1,1,2,2,1,0')
const holeText = ref('1:1,1:3,2:3,3:0')
const experimentName = ref('GridWorld Q-Learning 演示')
const persistResult = ref(true)
const comparisonPalette = ['#0f5bd8', '#d97706', '#0f766e', '#b42318']
const benchmarkEditor = reactive({
  name: '',
  description: '',
  teacher_note: '',
  average_reward: 0,
  best_reward: 0,
  success_rate: 0,
  stable_success_rate: 0,
})

const gridEnvConfig = reactive({
  size: 6,
  start: { row: 0, col: 0 },
  goal: { row: 5, col: 5 },
  rewards: {
    step_penalty: -1,
    goal_reward: 20,
    wall_penalty: -3,
    trap_penalty: -10,
  },
})

const cliffEnvConfig = reactive({
  rows: 4,
  cols: 12,
  start: { row: 3, col: 0 },
  goal: { row: 3, col: 11 },
  rewards: {
    step_penalty: -1,
    goal_reward: 0,
    wall_penalty: -1,
    cliff_penalty: -100,
  } satisfies CliffRewardConfig,
})

const windyEnvConfig = reactive({
  rows: 7,
  cols: 10,
  start: { row: 3, col: 0 },
  goal: { row: 3, col: 7 },
  rewards: {
    step_penalty: -1,
    goal_reward: 0,
    wall_penalty: -1,
  } satisfies WindyRewardConfig,
})

const frozenEnvConfig = reactive({
  rows: 4,
  cols: 4,
  start: { row: 0, col: 0 },
  goal: { row: 3, col: 3 },
  slip_probability: 0.2,
  rewards: {
    step_penalty: 0,
    goal_reward: 1,
    wall_penalty: 0,
    hole_penalty: 0,
  } satisfies FrozenLakeRewardConfig,
})

const training = reactive({
  episodes: 120,
  seed: 7,
  trace_frequency: 30,
})

const qLearningConfig = reactive<QLearningConfig>({
  learning_rate: 0.2,
  gamma: 0.92,
  epsilon_start: 1,
  epsilon_min: 0.05,
  epsilon_decay: 0.98,
  max_steps_per_episode: 80,
})

const sarsaConfig = reactive<SARSAConfig>({
  learning_rate: 0.2,
  gamma: 0.92,
  epsilon_start: 1,
  epsilon_min: 0.05,
  epsilon_decay: 0.98,
  max_steps_per_episode: 80,
})

const dqnConfig = reactive<DQNConfig>({
  learning_rate: 0.001,
  gamma: 0.95,
  epsilon_start: 1,
  epsilon_min: 0.05,
  epsilon_decay: 0.992,
  max_steps_per_episode: 80,
  batch_size: 32,
  replay_buffer_size: 2000,
  target_sync_interval: 40,
  warmup_steps: 80,
  hidden_dim: 64,
})

const reinforceConfig = reactive<ReinforceConfig>({
  learning_rate: 0.01,
  gamma: 0.95,
  max_steps_per_episode: 80,
  hidden_dim: 64,
})

function environmentLabel(environmentId: EnvironmentId) {
  if (environmentId === 'cliffwalking') {
    return 'CliffWalking'
  }
  if (environmentId === 'windygridworld') {
    return 'WindyGridWorld'
  }
  if (environmentId === 'frozenlake') {
    return 'FrozenLake'
  }
  return 'GridWorld'
}

function defaultCliffCells(rows: number, cols: number): GridPosition[] {
  if (rows < 1 || cols < 3) {
    return []
  }

  const cliffRow = rows - 1
  return Array.from({ length: cols - 2 }, (_, index) => ({
    row: cliffRow,
    col: index + 1,
  }))
}

function defaultWindStrengths(cols: number) {
  const base = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]
  if (cols <= base.length) {
    return base.slice(0, cols)
  }
  return [...base, ...Array.from({ length: cols - base.length }, () => 0)]
}

function defaultFrozenHoles(rows: number, cols: number): GridPosition[] {
  if (rows === 4 && cols === 4) {
    return [
      { row: 1, col: 1 },
      { row: 1, col: 3 },
      { row: 2, col: 3 },
      { row: 3, col: 0 },
    ]
  }
  return []
}

function resetGridEnvironment() {
  gridEnvConfig.size = 6
  gridEnvConfig.start = { row: 0, col: 0 }
  gridEnvConfig.goal = { row: 5, col: 5 }
  gridEnvConfig.rewards = {
    step_penalty: -1,
    goal_reward: 20,
    wall_penalty: -3,
    trap_penalty: -10,
  }
  obstacleText.value = '1:2,2:2,4:2,4:3'
  trapText.value = '2:4,4:1'
}

function resetCliffEnvironment() {
  cliffEnvConfig.rows = 4
  cliffEnvConfig.cols = 12
  cliffEnvConfig.start = { row: 3, col: 0 }
  cliffEnvConfig.goal = { row: 3, col: 11 }
  cliffEnvConfig.rewards = {
    step_penalty: -1,
    goal_reward: 0,
    wall_penalty: -1,
    cliff_penalty: -100,
  }
  cliffText.value = serializeCellList(defaultCliffCells(cliffEnvConfig.rows, cliffEnvConfig.cols))
}

function resetWindyEnvironment() {
  windyEnvConfig.rows = 7
  windyEnvConfig.cols = 10
  windyEnvConfig.start = { row: 3, col: 0 }
  windyEnvConfig.goal = { row: 3, col: 7 }
  windyEnvConfig.rewards = {
    step_penalty: -1,
    goal_reward: 0,
    wall_penalty: -1,
  }
  windStrengthText.value = serializeWindStrengths(defaultWindStrengths(windyEnvConfig.cols))
}

function resetFrozenEnvironment() {
  frozenEnvConfig.rows = 4
  frozenEnvConfig.cols = 4
  frozenEnvConfig.start = { row: 0, col: 0 }
  frozenEnvConfig.goal = { row: 3, col: 3 }
  frozenEnvConfig.slip_probability = 0.2
  frozenEnvConfig.rewards = {
    step_penalty: 0,
    goal_reward: 1,
    wall_penalty: 0,
    hole_penalty: 0,
  }
  holeText.value = serializeCellList(defaultFrozenHoles(frozenEnvConfig.rows, frozenEnvConfig.cols))
}

function nextExperimentName(algorithmId: AlgorithmId, environmentId: EnvironmentId = selectedEnvironment.value) {
  const environmentName = environmentLabel(environmentId)
  if (algorithmId === 'q_learning') {
    return `${environmentName} Q-Learning 演示`
  }
  if (algorithmId === 'sarsa') {
    return `${environmentName} SARSA 演示`
  }
  if (algorithmId === 'dqn') {
    return `${environmentName} DQN 演示`
  }
  return `${environmentName} REINFORCE 演示`
}

const heroMetrics = computed(() => [
  { label: '平台定位', value: '低代码强化学习实验室' },
  { label: '当前视角', value: roleLabel(viewerRole.value) },
  { label: '当前操作人', value: submittedBy.value || '未指定' },
])

const activeMetrics = computed(() => currentResult.value?.metrics ?? liveMetrics.value)
const rewardSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.reward })))
const epsilonSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.epsilon })))
const tdErrorSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.td_error })))

const summaryCards = computed(() => {
  if (currentResult.value) {
    return [
      { label: '平均奖励', value: currentResult.value.summary.average_reward.toFixed(3) },
      { label: '最佳奖励', value: currentResult.value.summary.best_reward.toFixed(3) },
      { label: '成功率', value: `${(currentResult.value.summary.success_rate * 100).toFixed(1)}%` },
      { label: '稳定窗口', value: `${(currentResult.value.summary.stable_success_rate * 100).toFixed(1)}%` },
    ]
  }

  if (!liveMetrics.value.length) {
    return []
  }

  const bestReward = Math.max(...liveMetrics.value.map((metric) => metric.reward))
  return [
    { label: '平均奖励', value: liveAverageReward.value.toFixed(3) },
    { label: '最佳奖励', value: bestReward.toFixed(3) },
    { label: '成功率', value: `${(liveSuccessRate.value * 100).toFixed(1)}%` },
    { label: '训练进度', value: `${liveCompletedEpisodes.value}/${liveTotalEpisodes.value} (${Math.round(liveProgress.value * 100)}%)` },
  ]
})

const latestTrace = computed(() => currentResult.value?.path_traces.at(-1) ?? liveLatestTrace.value)
const algorithmOptions = computed(() => catalog.value?.algorithms ?? [])
const environmentOptions = computed(() => catalog.value?.environments ?? [])
const assignmentOptions = computed(() =>
  (assignmentCatalog.value?.assignments ?? []).filter((assignment) => assignment.request.environment_id === selectedEnvironment.value),
)
const selectedAssignment = computed<AssignmentPreset | null>(
  () => assignmentOptions.value.find((assignment) => assignment.id === selectedAssignmentId.value) ?? null,
)
const benchmarkOptions = computed(() =>
  (benchmarkCatalog.value?.benchmarks ?? []).filter((benchmark) => benchmark.request.environment_id === selectedEnvironment.value),
)
const effectiveBenchmarkId = computed(() => {
  if (selectedBenchmarkId.value) {
    return selectedBenchmarkId.value
  }
  return selectedAssignment.value?.benchmark_id ?? ''
})
const selectedBenchmark = computed<BenchmarkPreset | null>(
  () => benchmarkOptions.value.find((benchmark) => benchmark.id === effectiveBenchmarkId.value) ?? null,
)
const selectedEditableBenchmark = computed<BenchmarkPreset | null>(() => {
  if (!selectedBenchmark.value || selectedBenchmark.value.is_builtin !== false) {
    return null
  }
  return selectedBenchmark.value
})
const activeRunId = computed(() => currentResult.value?.run_id ?? liveRunId.value)
const showResults = computed(() => Boolean(currentResult.value || liveMetrics.value.length || loading.value))
const showTeacherAnalytics = computed(() => viewerRole.value === 'teacher')
const finalPolicyGrid = computed(() => resultGrid(currentResult.value))
const policyGridColumnCount = computed(() => Math.max(finalPolicyGrid.value[0]?.length ?? 0, 0))
const widePolicyGrid = computed(() => policyGridColumnCount.value >= 10)
const canExportReport = computed(() => Boolean(currentResult.value) && !loading.value && !reportExporting.value)
const canPauseStream = computed(() => loading.value && streamState.value === 'running')
const canResumeStream = computed(() => loading.value && streamState.value === 'paused')
const canCancelStream = computed(() => loading.value && streamState.value !== 'cancelling')
const streamStateLabel = computed(() => {
  if (streamState.value === 'paused') {
    return '已暂停'
  }
  if (streamState.value === 'cancelling') {
    return '取消中'
  }
  return '训练中'
})
const compareRuns = computed(() => {
  const dedup = new Map<string, ExperimentResult>()
  if (currentResult.value) {
    dedup.set(currentResult.value.run_id, currentResult.value)
  }
  for (const runId of selectedComparisonIds.value) {
    const result = comparisonResults.value[runId]
    if (result) {
      dedup.set(result.run_id, result)
    }
  }
  return Array.from(dedup.values())
})
const replayRuns = computed(() => {
  if (compareRuns.value.length) {
    return compareRuns.value
  }
  return currentResult.value ? [currentResult.value] : []
})
const replaySourceResult = computed<ExperimentResult | null>(
  () => replayRuns.value.find((result) => result.run_id === replaySourceRunId.value) ?? replayRuns.value[0] ?? null,
)
const replayResultGrid = computed(() => resultGrid(replaySourceResult.value))
const replayGridColumnCount = computed(() => Math.max(replayResultGrid.value[0]?.length ?? 0, 0))
const wideReplayGrid = computed(() => replayGridColumnCount.value >= 10)
const replayTraceOptions = computed(() =>
  [...(replaySourceResult.value?.path_traces ?? [])].sort((left, right) => right.episode - left.episode),
)
const replayActiveTrace = computed<PathTrace | null>(
  () => replayTraceOptions.value.find((trace) => trace.episode === replayTraceEpisode.value) ?? replayTraceOptions.value[0] ?? null,
)
const replayPathLength = computed(() => replayActiveTrace.value?.path.length ?? 0)
const replayMaxStepIndex = computed(() => Math.max(replayPathLength.value - 1, 0))
const replayCurrentCell = computed(() => {
  if (!replayActiveTrace.value || !replayPathLength.value) {
    return null
  }
  return replayActiveTrace.value.path[Math.min(replayStepIndex.value, replayMaxStepIndex.value)]
})
const canStepReplayBackward = computed(() => replayPathLength.value > 0 && replayStepIndex.value > 0)
const canStepReplayForward = computed(
  () => replayPathLength.value > 0 && replayStepIndex.value < replayMaxStepIndex.value,
)
const replayProgressLabel = computed(() => {
  if (!replayPathLength.value) {
    return '当前未加载采样路径。'
  }
  return `回放步数 ${replayStepIndex.value + 1}/${replayPathLength.value}`
})
const compareRewardSeries = computed<ComparisonSeries[]>(() =>
  compareRuns.value.map((result, index) => ({
    runId: result.run_id,
    label: comparisonLabel(result, currentResult.value?.run_id === result.run_id),
    color: comparisonPalette[index % comparisonPalette.length],
    points: result.metrics.map((metric) => ({ x: metric.episode, y: metric.reward })),
  })),
)
const compareSuccessSeries = computed<ComparisonSeries[]>(() =>
  compareRuns.value.map((result, index) => ({
    runId: result.run_id,
    label: comparisonLabel(result, currentResult.value?.run_id === result.run_id),
    color: comparisonPalette[index % comparisonPalette.length],
    points: result.metrics.map((metric) => ({ x: metric.episode, y: metric.success ? 1 : 0 })),
  })),
)
const compareSummaryRows = computed(() =>
  compareRuns.value.map((result) => ({
    runId: result.run_id,
    label: comparisonLabel(result, currentResult.value?.run_id === result.run_id),
    algorithm: algorithmLabel(result.request.algorithm_id),
    averageReward: result.summary.average_reward.toFixed(3),
    bestReward: result.summary.best_reward.toFixed(3),
    successRate: `${(result.summary.success_rate * 100).toFixed(1)}%`,
    episodes: result.request.training.episodes,
  })),
)
const comparisonReady = computed(() => compareRuns.value.length >= 2)
const teacherAnalyticsCards = computed(() => {
  if (!analytics.value) {
    return []
  }

  return [
    { label: '实验总次数', value: `${analytics.value.total_runs}` },
    { label: '学生实验次数', value: `${analytics.value.student_runs}` },
    {
      label: analytics.value.benchmark ? '基准通过率' : '教师实验次数',
      value: analytics.value.benchmark
        ? `${(analytics.value.benchmark.pass_rate * 100).toFixed(1)}%`
        : `${analytics.value.teacher_runs}`,
    },
    { label: '平均成功率', value: `${(analytics.value.average_success_rate * 100).toFixed(1)}%` },
  ]
})
const effectiveAnalyticsBenchmarkId = computed(() => {
  if (selectedAssignment.value?.benchmark_id) {
    return selectedAssignment.value.benchmark_id
  }
  return selectedBenchmarkId.value || null
})
const benchmarkEvaluation = computed<BenchmarkEvaluation | null>(() => {
  if (!currentResult.value || !selectedBenchmark.value) {
    return null
  }

  const result = currentResult.value
  const benchmark = selectedBenchmark.value

  const compatibilityChecks: BenchmarkCheckRow[] = [
    {
      label: '算法一致性',
      passed: result.request.algorithm_id === benchmark.request.algorithm_id,
      expected: algorithmLabel(benchmark.request.algorithm_id),
      actual: algorithmLabel(result.request.algorithm_id),
      helpText: '教师基准应在相同算法类型下进行比较。',
    },
    {
      label: '环境配置一致性',
      passed: sameEnvironmentConfig(result.request.env_config, benchmark.request.env_config),
      expected: environmentSignatureLabel(benchmark.request.env_config),
      actual: environmentSignatureLabel(result.request.env_config),
      helpText: '环境类型、关键单元分布和奖励设置不同，都会改变任务难度，削弱直接比较的意义。',
    },
    {
      label: '训练轮次要求',
      passed: result.request.training.episodes >= benchmark.request.training.episodes,
      expected: `>= ${benchmark.request.training.episodes}`,
      actual: `${result.request.training.episodes}`,
      helpText: '训练轮次不足时，不宜与教师基准直接比较。',
    },
  ]

  const thresholdChecks = benchmark.thresholds.map((threshold) => {
    const actualValue = getSummaryMetric(currentResult.value as ExperimentResult, threshold.metric_id)
    return {
      label: threshold.label,
      passed: actualValue >= threshold.min_value,
      expected: `>= ${formatBenchmarkMetric(threshold.metric_id, threshold.min_value)}`,
      actual: formatBenchmarkMetric(threshold.metric_id, actualValue),
      helpText: threshold.help_text,
    }
  })

  return {
    passed: [...compatibilityChecks, ...thresholdChecks].every((check) => check.passed),
    compatibilityChecks,
    thresholdChecks,
  }
})

const benchmarkOverview = computed<BenchmarkOverview | null>(() => {
  if (!benchmarkEvaluation.value) {
    return null
  }

  const compatibilityFailures = benchmarkEvaluation.value.compatibilityChecks.filter((check) => !check.passed)
  const thresholdFailures = benchmarkEvaluation.value.thresholdChecks.filter((check) => !check.passed)
  const totalFailures = compatibilityFailures.length + thresholdFailures.length

  if (benchmarkEvaluation.value.passed) {
    return {
      passed: true,
      headline: '当前结果已达到教师基准',
      message: '所有前置条件与指标阈值均已通过，可直接作为当前环境与算法配置下的参考结果。',
    }
  }

  if (compatibilityFailures.length) {
    return {
      passed: false,
      headline: '当前结果暂不适合直接对标',
      message: `请先解决“${compatibilityFailures[0].label}”。只要比较前提不一致，后续阈值是否通过都不具备直接对照意义。`,
    }
  }

  return {
    passed: false,
    headline: '当前结果接近基准，但仍需补强',
    message: `还有 ${totalFailures} 项指标未达标，优先关注“${thresholdFailures[0]?.label ?? '关键指标'}”。`,
  }
})

const benchmarkOverviewStats = computed<BenchmarkOverviewStat[]>(() => {
  if (!benchmarkEvaluation.value) {
    return []
  }

  const allChecks = [...benchmarkEvaluation.value.compatibilityChecks, ...benchmarkEvaluation.value.thresholdChecks]
  const failedChecks = allChecks.filter((check) => !check.passed)

  return [
    {
      label: '通过检查',
      value: `${allChecks.length - failedChecks.length}/${allChecks.length}`,
      tone: failedChecks.length ? 'neutral' : 'pass',
    },
    {
      label: '待改进项',
      value: `${failedChecks.length}`,
      tone: failedChecks.length ? 'fail' : 'pass',
    },
    {
      label: '首要问题',
      value: failedChecks[0]?.label ?? '全部达标',
      tone: failedChecks.length ? 'neutral' : 'pass',
    },
  ]
})

const benchmarkMetricCards = computed<BenchmarkMetricCard[]>(() => {
  if (!currentResult.value || !selectedBenchmark.value) {
    return []
  }

  return selectedBenchmark.value.thresholds.map((threshold) => {
    const actualValue = getSummaryMetric(currentResult.value as ExperimentResult, threshold.metric_id)
    const delta = actualValue - threshold.min_value
    const passed = delta >= 0
    const deltaText =
      Math.abs(delta) < 1e-9
        ? '刚好达到目标'
        : passed
          ? `超出目标 ${formatBenchmarkGap(threshold.metric_id, delta)}`
          : `距离目标还差 ${formatBenchmarkGap(threshold.metric_id, delta)}`

    return {
      label: threshold.label,
      passed,
      actual: formatBenchmarkMetric(threshold.metric_id, actualValue),
      expected: formatBenchmarkMetric(threshold.metric_id, threshold.min_value),
      deltaText,
      helpText: threshold.help_text,
    }
  })
})

watch(
  () => gridEnvConfig.size,
  (size) => {
    gridEnvConfig.start.row = clampCellCoordinate(gridEnvConfig.start.row, size - 1)
    gridEnvConfig.start.col = clampCellCoordinate(gridEnvConfig.start.col, size - 1)
    gridEnvConfig.goal.row = clampCellCoordinate(gridEnvConfig.goal.row, size - 1)
    gridEnvConfig.goal.col = clampCellCoordinate(gridEnvConfig.goal.col, size - 1)
    if (size !== 6) {
      obstacleText.value = ''
      trapText.value = ''
    }
  },
)

watch(
  [selectedAlgorithm, selectedEnvironment],
  ([algorithmId, environmentId]) => {
    experimentName.value = nextExperimentName(algorithmId, environmentId)
  },
)

watch(
  [() => cliffEnvConfig.rows, () => cliffEnvConfig.cols],
  ([rows, cols]) => {
    cliffEnvConfig.start.row = clampCellCoordinate(cliffEnvConfig.start.row, rows - 1)
    cliffEnvConfig.start.col = clampCellCoordinate(cliffEnvConfig.start.col, cols - 1)
    cliffEnvConfig.goal.row = clampCellCoordinate(cliffEnvConfig.goal.row, rows - 1)
    cliffEnvConfig.goal.col = clampCellCoordinate(cliffEnvConfig.goal.col, cols - 1)
    cliffText.value = serializeCellList(defaultCliffCells(rows, cols))
  },
)

watch(
  () => windyEnvConfig.rows,
  (rows) => {
    windyEnvConfig.start.row = clampCellCoordinate(windyEnvConfig.start.row, rows - 1)
    windyEnvConfig.goal.row = clampCellCoordinate(windyEnvConfig.goal.row, rows - 1)
  },
)

watch(
  () => windyEnvConfig.cols,
  (cols) => {
    windyEnvConfig.start.col = clampCellCoordinate(windyEnvConfig.start.col, cols - 1)
    windyEnvConfig.goal.col = clampCellCoordinate(windyEnvConfig.goal.col, cols - 1)
    windStrengthText.value = serializeWindStrengths(defaultWindStrengths(cols))
  },
)

watch(
  [() => frozenEnvConfig.rows, () => frozenEnvConfig.cols],
  ([rows, cols]) => {
    frozenEnvConfig.start.row = clampCellCoordinate(frozenEnvConfig.start.row, rows - 1)
    frozenEnvConfig.start.col = clampCellCoordinate(frozenEnvConfig.start.col, cols - 1)
    frozenEnvConfig.goal.row = clampCellCoordinate(frozenEnvConfig.goal.row, rows - 1)
    frozenEnvConfig.goal.col = clampCellCoordinate(frozenEnvConfig.goal.col, cols - 1)
    holeText.value = serializeCellList(defaultFrozenHoles(rows, cols))
  },
)

watch(selectedEnvironment, () => {
  if (!assignmentOptions.value.some((assignment) => assignment.id === selectedAssignmentId.value)) {
    selectedAssignmentId.value = ''
  }
  if (!benchmarkOptions.value.some((benchmark) => benchmark.id === selectedBenchmarkId.value)) {
    selectedBenchmarkId.value = ''
  }
})

watch(
  selectedBenchmark,
  (benchmark) => {
    syncBenchmarkEditor(benchmark)
  },
  { immediate: true },
)

watch([selectedEnvironment, selectedAlgorithm, currentResult], () => {
  if (!selectedBenchmark.value) {
    syncBenchmarkEditor(null)
  }
})

watch(viewerRole, (role, previousRole) => {
  const currentName = submittedBy.value.trim()
  const wasDefaultStudent = currentName === '' || currentName === '学生A'
  const wasDefaultTeacher = currentName === '课程教师'
  if ((previousRole === 'student' && wasDefaultStudent) || (previousRole === 'teacher' && wasDefaultTeacher)) {
    submittedBy.value = role === 'teacher' ? '课程教师' : '学生A'
  }
})

onMounted(async () => {
  await Promise.all([loadCatalog(), loadAssignments(), loadBenchmarks(), loadHistory()])
})

onBeforeUnmount(() => {
  stopReplay()
})

watch(
  selectedComparisonIds,
  async (runIds) => {
    await Promise.all(runIds.map((runId) => ensureComparisonRun(runId)))
  },
  { deep: true },
)

watch(
  replayRuns,
  (runs) => {
    if (!runs.length) {
      replaySourceRunId.value = ''
      replayTraceEpisode.value = 0
      replayStepIndex.value = 0
      stopReplay()
      return
    }

    if (!runs.some((result) => result.run_id === replaySourceRunId.value)) {
      replaySourceRunId.value = runs[0].run_id
    }
  },
  { deep: true, immediate: true },
)

watch(
  replaySourceResult,
  (result) => {
    if (!result) {
      replayTraceEpisode.value = 0
      replayStepIndex.value = 0
      stopReplay()
      return
    }

    const latestTrace = [...result.path_traces].sort((left, right) => right.episode - left.episode)[0]
    replayTraceEpisode.value = latestTrace?.episode ?? 0
    replayStepIndex.value = 0
    stopReplay()
  },
  { immediate: true },
)

watch(
  replayActiveTrace,
  () => {
    replayStepIndex.value = 0
    stopReplay()
  },
  { immediate: true },
)

watch(
  [viewerRole, selectedBenchmarkId, selectedAssignmentId],
  async ([role]) => {
    if (role === 'teacher') {
      await loadAnalytics()
      return
    }
    analytics.value = null
  },
  { deep: true },
)

async function loadCatalog() {
  try {
    catalog.value = await getCatalog()
    const availableEnvironmentIds = catalog.value.environments.map((environment) => environment.id as EnvironmentId)
    if (availableEnvironmentIds.length && !availableEnvironmentIds.includes(selectedEnvironment.value)) {
      selectedEnvironment.value = availableEnvironmentIds[0]
    }
  } catch (error) {
    console.error(error)
  }
}

async function loadAssignments() {
  try {
    assignmentCatalog.value = await getAssignments()
    if (!assignmentCatalog.value.assignments.some((assignment) => assignment.id === selectedAssignmentId.value)) {
      selectedAssignmentId.value = ''
    }
  } catch (error) {
    console.error(error)
    assignmentCatalog.value = { assignments: [] }
    selectedAssignmentId.value = ''
  }
}

async function loadBenchmarks() {
  try {
    benchmarkCatalog.value = await getBenchmarks()
    if (selectedBenchmarkId.value && !benchmarkCatalog.value.benchmarks.some((benchmark) => benchmark.id === selectedBenchmarkId.value)) {
      selectedBenchmarkId.value = ''
    }
  } catch (error) {
    console.error(error)
    benchmarkCatalog.value = { benchmarks: [] }
    selectedBenchmarkId.value = ''
  }
}

async function loadHistory() {
  try {
    history.value = await getHistory()
    selectedComparisonIds.value = selectedComparisonIds.value.filter((runId) =>
      history.value.some((item) => item.run_id === runId),
    )
  } catch (error) {
    console.error(error)
    history.value = []
  }
}

async function loadAnalytics() {
  analyticsLoading.value = true
  try {
    analytics.value = await getClassroomAnalytics(
      effectiveAnalyticsBenchmarkId.value,
      selectedAssignmentId.value || null,
    )
  } catch (error) {
    console.error(error)
    analytics.value = null
  } finally {
    analyticsLoading.value = false
  }
}

function algorithmLabel(algorithmId: AlgorithmId) {
  if (algorithmId === 'q_learning') {
    return 'Q-Learning'
  }
  if (algorithmId === 'sarsa') {
    return 'SARSA'
  }
  if (algorithmId === 'dqn') {
    return 'DQN'
  }
  return 'REINFORCE'
}

function roleLabel(role: SubmissionRole) {
  return role === 'teacher' ? '教师视角' : '学生视角'
}

function isGridWorldConfig(config: EnvironmentConfig): config is GridWorldConfig {
  return config.environment_id === 'gridworld'
}

function isCliffWalkingConfig(config: EnvironmentConfig): config is CliffWalkingConfig {
  return config.environment_id === 'cliffwalking'
}

function isWindyGridWorldConfig(config: EnvironmentConfig): config is WindyGridWorldConfig {
  return config.environment_id === 'windygridworld'
}

function isFrozenLakeConfig(config: EnvironmentConfig): config is FrozenLakeConfig {
  return config.environment_id === 'frozenlake'
}

function environmentShapeLabel(config: EnvironmentConfig) {
  return `${config.rows} x ${config.cols}`
}

function sortCellList(cells: GridPosition[]) {
  return [...cells].sort((left, right) => (left.row - right.row) || (left.col - right.col))
}

function normalizeEnvironmentConfig(config: EnvironmentConfig) {
  if (isGridWorldConfig(config)) {
    return {
      environment_id: config.environment_id,
      rows: config.rows,
      cols: config.cols,
      start: config.start,
      goal: config.goal,
      obstacles: sortCellList(config.obstacles),
      traps: sortCellList(config.traps),
      rewards: config.rewards,
    }
  }
  if (isCliffWalkingConfig(config)) {
    return {
      environment_id: config.environment_id,
      rows: config.rows,
      cols: config.cols,
      start: config.start,
      goal: config.goal,
      cliffs: sortCellList(config.cliffs),
      rewards: config.rewards,
    }
  }
  if (isFrozenLakeConfig(config)) {
    return {
      environment_id: config.environment_id,
      rows: config.rows,
      cols: config.cols,
      start: config.start,
      goal: config.goal,
      holes: sortCellList(config.holes),
      slip_probability: config.slip_probability,
      rewards: config.rewards,
    }
  }
  return {
    environment_id: config.environment_id,
    rows: config.rows,
    cols: config.cols,
    start: config.start,
    goal: config.goal,
    wind_strengths: [...config.wind_strengths],
    rewards: config.rewards,
  }
}

function sameEnvironmentConfig(left: EnvironmentConfig, right: EnvironmentConfig) {
  return JSON.stringify(normalizeEnvironmentConfig(left)) === JSON.stringify(normalizeEnvironmentConfig(right))
}

function environmentSignatureLabel(config: EnvironmentConfig) {
  const prefix = `${environmentLabel(config.environment_id)} | ${environmentShapeLabel(config)}`
  if (isFrozenLakeConfig(config)) {
    return `${prefix} | 冰洞 ${config.holes.length} | 滑移 ${config.slip_probability}`
  }
  if (isGridWorldConfig(config)) {
    return `${prefix} | 障碍 ${config.obstacles.length} | 陷阱 ${config.traps.length}`
  }
  if (isCliffWalkingConfig(config)) {
    return `${prefix} | 悬崖 ${config.cliffs.length}`
  }
  return `${prefix} | 风列 [${config.wind_strengths.join(', ')}]`
}

function clampCellCoordinate(value: number, max: number) {
  return Math.min(Math.max(value, 0), Math.max(max, 0))
}

function resultGrid(result: ExperimentResult | null) {
  if (!result) {
    return []
  }
  if (result.environment_view?.view_type === 'grid') {
    return result.environment_view.cells
  }
  return result.policy_grid
}

function parseCellList(input: string): GridPosition[] {
  if (!input.trim()) {
    return []
  }

  return input
    .split(',')
    .map((pair) => pair.trim())
    .filter(Boolean)
    .map((pair) => {
      const [rowValue, colValue] = pair.split(':')
      const row = Number(rowValue)
      const col = Number(colValue)
      if (!Number.isInteger(row) || !Number.isInteger(col)) {
        throw new Error(`单元格格式错误：${pair}。请使用 1:2,2:2 这样的 row:col 格式。`)
      }
      return { row, col }
    })
}

function parseWindStrengths(input: string, expectedCols: number) {
  const strengths = input
    .split(',')
    .map((value) => value.trim())
    .filter(Boolean)
    .map((value) => {
      const parsed = Number(value)
      if (!Number.isInteger(parsed) || parsed < 0) {
        throw new Error(`风力强度格式错误：${value}。请使用 0,0,1,2 这样的非负整数序列。`)
      }
      return parsed
    })

  if (strengths.length !== expectedCols) {
    throw new Error(`风力强度数量应与列数一致。当前列数为 ${expectedCols}，但输入了 ${strengths.length} 个值。`)
  }

  return strengths
}

function serializeCellList(cells: GridPosition[]) {
  return cells.map((cell) => `${cell.row}:${cell.col}`).join(',')
}

function serializeWindStrengths(strengths: number[]) {
  return strengths.join(',')
}

function defaultBenchmarkName() {
  return `${environmentLabel(selectedEnvironment.value)} ${algorithmLabel(selectedAlgorithm.value)} 教师基准`
}

function syncBenchmarkEditor(benchmark: BenchmarkPreset | null) {
  if (benchmark) {
    benchmarkEditor.name = benchmark.name
    benchmarkEditor.description = benchmark.description
    benchmarkEditor.teacher_note = benchmark.teacher_note
    benchmarkEditor.average_reward =
      benchmark.thresholds.find((threshold) => threshold.metric_id === 'average_reward')?.min_value ?? 0
    benchmarkEditor.best_reward =
      benchmark.thresholds.find((threshold) => threshold.metric_id === 'best_reward')?.min_value ?? 0
    benchmarkEditor.success_rate =
      benchmark.thresholds.find((threshold) => threshold.metric_id === 'success_rate')?.min_value ?? 0
    benchmarkEditor.stable_success_rate =
      benchmark.thresholds.find((threshold) => threshold.metric_id === 'stable_success_rate')?.min_value ?? 0
    return
  }

  benchmarkEditor.name = defaultBenchmarkName()
  benchmarkEditor.description = `用于 ${environmentLabel(selectedEnvironment.value)} 环境下 ${algorithmLabel(selectedAlgorithm.value)} 实验的教师自定义基准。`
  benchmarkEditor.teacher_note = '请结合课堂目标调整阈值，并在同算法、同环境配置下进行对比。'
  benchmarkEditor.average_reward = currentResult.value?.summary.average_reward ?? 0
  benchmarkEditor.best_reward = currentResult.value?.summary.best_reward ?? 0
  benchmarkEditor.success_rate = currentResult.value?.summary.success_rate ?? 0
  benchmarkEditor.stable_success_rate = currentResult.value?.summary.stable_success_rate ?? 0
}

function comparisonLabel(result: ExperimentResult, isCurrent = false) {
  const prefix = isCurrent ? '当前实验' : result.request.name
  return `${prefix} (${algorithmLabel(result.request.algorithm_id)})`
}

function formatBenchmarkMetric(metricId: BenchmarkThreshold['metric_id'], value: number) {
  if (metricId === 'success_rate' || metricId === 'stable_success_rate') {
    return `${(value * 100).toFixed(1)}%`
  }
  return value.toFixed(3)
}

function formatBenchmarkGap(metricId: BenchmarkThreshold['metric_id'], value: number) {
  const magnitude = Math.abs(value)
  if (metricId === 'success_rate' || metricId === 'stable_success_rate') {
    return `${(magnitude * 100).toFixed(1)} 个百分点`
  }
  return magnitude.toFixed(3)
}

function formatTimestamp(timestamp: string) {
  return new Date(timestamp).toLocaleString()
}

function ensureFiniteNumber(value: number, label: string) {
  if (!Number.isFinite(value)) {
    throw new Error(`${label} 不能为空，请输入有效数字。`)
  }
  return value
}

function ensureMinTextLength(value: string, minLength: number, label: string) {
  const trimmed = value.trim()
  if (trimmed.length < minLength) {
    throw new Error(`${label} 至少需要 ${minLength} 个字符。`)
  }
  return trimmed
}

function ensureProbability(value: number, label: string) {
  let normalized = ensureFiniteNumber(value, label)
  if (normalized > 1 && normalized <= 100) {
    normalized /= 100
  }
  if (normalized < 0 || normalized > 1) {
    throw new Error(`${label} 需要在 0 到 1 之间，也可以直接输入 60 代表 60%。`)
  }
  return normalized
}

function getSummaryMetric(result: ExperimentResult, metricId: BenchmarkThreshold['metric_id']) {
  switch (metricId) {
    case 'average_reward':
      return result.summary.average_reward
    case 'best_reward':
      return result.summary.best_reward
    case 'success_rate':
      return result.summary.success_rate
    case 'stable_success_rate':
      return result.summary.stable_success_rate
  }
}

function isComparisonSelected(runId: string) {
  return selectedComparisonIds.value.includes(runId)
}

function toggleComparison(runId: string) {
  if (isComparisonSelected(runId)) {
    selectedComparisonIds.value = selectedComparisonIds.value.filter((id) => id !== runId)
    return
  }

  if (selectedComparisonIds.value.length >= 3) {
    errorMessage.value = '最多只能同时对比三条已保存的实验记录。'
    return
  }

  selectedComparisonIds.value = [...selectedComparisonIds.value, runId]
}

async function ensureComparisonRun(runId: string) {
  if (comparisonResults.value[runId] || comparisonLoadingIds.value.includes(runId)) {
    return
  }

  comparisonLoadingIds.value = [...comparisonLoadingIds.value, runId]
  try {
    const result = await getExperiment(runId)
    comparisonResults.value = {
      ...comparisonResults.value,
      [runId]: result,
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '加载对比实验失败。'
  } finally {
    comparisonLoadingIds.value = comparisonLoadingIds.value.filter((id) => id !== runId)
  }
}

function loadDemoPreset() {
  selectedAssignmentId.value = ''
  selectedBenchmarkId.value = ''
  experimentName.value = nextExperimentName(selectedAlgorithm.value, selectedEnvironment.value)
  if (selectedEnvironment.value === 'gridworld') {
    resetGridEnvironment()
  } else if (selectedEnvironment.value === 'cliffwalking') {
    resetCliffEnvironment()
  } else if (selectedEnvironment.value === 'frozenlake') {
    resetFrozenEnvironment()
  } else {
    resetWindyEnvironment()
  }
  Object.assign(qLearningConfig, {
    learning_rate: 0.2,
    gamma: 0.92,
    epsilon_start: 1,
    epsilon_min: 0.05,
    epsilon_decay: 0.98,
    max_steps_per_episode: 80,
  })
  Object.assign(sarsaConfig, {
    learning_rate: 0.2,
    gamma: 0.92,
    epsilon_start: 1,
    epsilon_min: 0.05,
    epsilon_decay: 0.98,
    max_steps_per_episode: 80,
  })
  Object.assign(dqnConfig, {
    learning_rate: 0.001,
    gamma: 0.95,
    epsilon_start: 1,
    epsilon_min: 0.05,
    epsilon_decay: 0.992,
    max_steps_per_episode: 80,
    batch_size: 32,
    replay_buffer_size: 2000,
    target_sync_interval: 40,
    warmup_steps: 80,
    hidden_dim: 64,
  })
  Object.assign(reinforceConfig, {
    learning_rate: 0.01,
    gamma: 0.95,
    max_steps_per_episode: 80,
    hidden_dim: 64,
  })
  Object.assign(training, {
    episodes: 120,
    seed: 7,
    trace_frequency: 30,
  })
}

function applyExperimentRequestPreset(request: ExperimentRequest) {
  selectedEnvironment.value = request.environment_id
  selectedAlgorithm.value = request.algorithm_id
  experimentName.value = request.name
  if (isGridWorldConfig(request.env_config)) {
    gridEnvConfig.size = request.env_config.size
    gridEnvConfig.start = { ...request.env_config.start }
    gridEnvConfig.goal = { ...request.env_config.goal }
    gridEnvConfig.rewards = { ...request.env_config.rewards }
    obstacleText.value = serializeCellList(request.env_config.obstacles)
    trapText.value = serializeCellList(request.env_config.traps)
  } else if (isCliffWalkingConfig(request.env_config)) {
    cliffEnvConfig.rows = request.env_config.rows
    cliffEnvConfig.cols = request.env_config.cols
    cliffEnvConfig.start = { ...request.env_config.start }
    cliffEnvConfig.goal = { ...request.env_config.goal }
    cliffEnvConfig.rewards = { ...request.env_config.rewards }
    cliffText.value = serializeCellList(request.env_config.cliffs)
  } else if (isWindyGridWorldConfig(request.env_config)) {
    windyEnvConfig.rows = request.env_config.rows
    windyEnvConfig.cols = request.env_config.cols
    windyEnvConfig.start = { ...request.env_config.start }
    windyEnvConfig.goal = { ...request.env_config.goal }
    windyEnvConfig.rewards = { ...request.env_config.rewards }
    windStrengthText.value = serializeWindStrengths(request.env_config.wind_strengths)
  } else if (isFrozenLakeConfig(request.env_config)) {
    frozenEnvConfig.rows = request.env_config.rows
    frozenEnvConfig.cols = request.env_config.cols
    frozenEnvConfig.start = { ...request.env_config.start }
    frozenEnvConfig.goal = { ...request.env_config.goal }
    frozenEnvConfig.slip_probability = request.env_config.slip_probability
    frozenEnvConfig.rewards = { ...request.env_config.rewards }
    holeText.value = serializeCellList(request.env_config.holes)
  }

  training.episodes = request.training.episodes
  training.seed = request.training.seed
  training.trace_frequency = request.training.trace_frequency

  if (request.algorithm_id === 'q_learning') {
    Object.assign(qLearningConfig, request.algorithm_config)
  } else if (request.algorithm_id === 'sarsa') {
    Object.assign(sarsaConfig, request.algorithm_config)
  } else if (request.algorithm_id === 'dqn') {
    Object.assign(dqnConfig, request.algorithm_config)
  } else {
    Object.assign(reinforceConfig, request.algorithm_config)
  }
}

function buildBenchmarkDraft(): BenchmarkDraft {
  const request = buildPayload()
  const benchmarkName = ensureMinTextLength(
    benchmarkEditor.name.trim() || defaultBenchmarkName(),
    3,
    '基准名称',
  )
  const averageReward = ensureFiniteNumber(benchmarkEditor.average_reward, '平均奖励阈值')
  const bestReward = ensureFiniteNumber(benchmarkEditor.best_reward, '最佳奖励阈值')
  const successRate = ensureProbability(benchmarkEditor.success_rate, '成功率阈值')
  const stableSuccessRate = ensureProbability(benchmarkEditor.stable_success_rate, '稳定窗口阈值')

  return {
    name: benchmarkName,
    description: benchmarkEditor.description.trim() || `${environmentLabel(selectedEnvironment.value)} 教师基准。`,
    teacher_note: benchmarkEditor.teacher_note.trim() || '请在相同环境与算法配置下使用该教师基准。',
    request: {
      ...request,
      name: benchmarkName,
      submitted_by: '课程教师',
      submission_role: 'teacher',
      assignment_id: null,
      assignment_title: null,
      persist_result: false,
    },
    thresholds: [
      {
        metric_id: 'average_reward',
        label: '平均奖励',
        min_value: averageReward,
        help_text: '整体平均奖励应达到教师设定的参考水平。',
      },
      {
        metric_id: 'best_reward',
        label: '最佳奖励',
        min_value: bestReward,
        help_text: '训练过程中至少应出现一次较高质量的成功轨迹。',
      },
      {
        metric_id: 'success_rate',
        label: '成功率',
        min_value: successRate,
        help_text: '成功回合占比应达到课堂要求。',
      },
      {
        metric_id: 'stable_success_rate',
        label: '稳定窗口',
        min_value: stableSuccessRate,
        help_text: '训练后期的成功率应体现相对稳定的策略表现。',
      },
    ],
  }
}

async function saveBenchmarkEditor() {
  benchmarkSaving.value = true
  errorMessage.value = ''
  const editingBenchmarkId = selectedEditableBenchmark.value?.id ?? null
  try {
    const draft = buildBenchmarkDraft()
    const saved = editingBenchmarkId
      ? await updateBenchmark(editingBenchmarkId, draft)
      : await createBenchmark(draft)
    await loadBenchmarks()
    selectedBenchmarkId.value = saved.id
    streamNotice.value = editingBenchmarkId ? '教师基准已更新。' : '教师基准已保存。'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '保存教师基准失败。'
  } finally {
    benchmarkSaving.value = false
  }
}

async function removeSelectedBenchmark() {
  if (!selectedEditableBenchmark.value) {
    return
  }

  benchmarkDeleting.value = true
  errorMessage.value = ''
  try {
    await deleteBenchmark(selectedEditableBenchmark.value.id)
    selectedBenchmarkId.value = ''
    await loadBenchmarks()
    streamNotice.value = '教师基准已删除。'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '删除教师基准失败。'
  } finally {
    benchmarkDeleting.value = false
  }
}

function applySelectedBenchmark() {
  if (!selectedBenchmark.value) {
    return
  }

  applyExperimentRequestPreset(selectedBenchmark.value.request)
  streamNotice.value = `已加载教师基准模板：${selectedBenchmark.value.name}`
}

function applySelectedAssignment() {
  if (!selectedAssignment.value) {
    return
  }

  selectedBenchmarkId.value = selectedAssignment.value.benchmark_id ?? ''
  applyExperimentRequestPreset(selectedAssignment.value.request)
  experimentName.value = selectedAssignment.value.request.name
  streamNotice.value = `已加载作业模板：${selectedAssignment.value.title}`
}

async function exportExperimentReport() {
  if (!currentResult.value) {
    return
  }

  reportExporting.value = true
  errorMessage.value = ''

  try {
    const blob = await renderExperimentReport({
      result: currentResult.value,
      benchmark_id: selectedBenchmark.value?.id ?? null,
    })
    const reportUrl = URL.createObjectURL(blob)
    const benchmarkSuffix = selectedBenchmark.value ? `-${selectedBenchmark.value.id}` : ''
    const link = document.createElement('a')
    link.href = reportUrl
    link.download = `${currentResult.value.run_id}${benchmarkSuffix}-report.md`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(reportUrl)
    streamNotice.value = '实验报告已导出为 Markdown 文件。'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '导出实验报告失败。'
  } finally {
    reportExporting.value = false
  }
}

function stopReplay() {
  if (replayTimer !== null) {
    window.clearInterval(replayTimer)
    replayTimer = null
  }
  replayPlaying.value = false
}

function startReplay() {
  if (!replayActiveTrace.value || replayPathLength.value <= 1) {
    replayPlaying.value = false
    return
  }

  if (replayStepIndex.value >= replayMaxStepIndex.value) {
    replayStepIndex.value = 0
  }

  stopReplay()
  replayPlaying.value = true
  replayTimer = window.setInterval(() => {
    if (replayStepIndex.value >= replayMaxStepIndex.value) {
      stopReplay()
      return
    }
    replayStepIndex.value += 1
  }, 550)
}

function toggleReplay() {
  if (replayPlaying.value) {
    stopReplay()
    return
  }
  startReplay()
}

function stepReplay(delta: -1 | 1) {
  if (!replayActiveTrace.value) {
    return
  }
  stopReplay()
  replayStepIndex.value = Math.min(Math.max(replayStepIndex.value + delta, 0), replayMaxStepIndex.value)
}

function resetReplay() {
  stopReplay()
  replayStepIndex.value = 0
}

function formatReplayRunLabel(result: ExperimentResult) {
  const scope = result.request.assignment_title ?? result.request.name
  return `${scope} | ${result.request.submitted_by} | ${algorithmLabel(result.request.algorithm_id)}`
}

function formatGridPosition(cell: GridPosition) {
  return `(${cell.row}, ${cell.col})`
}

function formatReplayPath() {
  if (!replayActiveTrace.value || !replayPathLength.value) {
    return '当前没有可供回放的采样路径。'
  }

  return replayActiveTrace.value.path
    .map(
      (cell, index) =>
        `${index === replayStepIndex.value ? '[' : ''}(${cell.row}, ${cell.col})${index === replayStepIndex.value ? ']' : ''}`,
    )
    .join(' -> ')
}

function buildEnvironmentConfig(): EnvironmentConfig {
  if (selectedEnvironment.value === 'cliffwalking') {
    const config: CliffWalkingConfig = {
      environment_id: 'cliffwalking',
      rows: cliffEnvConfig.rows,
      cols: cliffEnvConfig.cols,
      start: { ...cliffEnvConfig.start },
      goal: { ...cliffEnvConfig.goal },
      cliffs: parseCellList(cliffText.value),
      rewards: { ...cliffEnvConfig.rewards },
    }
    return config
  }

  if (selectedEnvironment.value === 'windygridworld') {
    const config: WindyGridWorldConfig = {
      environment_id: 'windygridworld',
      rows: windyEnvConfig.rows,
      cols: windyEnvConfig.cols,
      start: { ...windyEnvConfig.start },
      goal: { ...windyEnvConfig.goal },
      wind_strengths: parseWindStrengths(windStrengthText.value, windyEnvConfig.cols),
      rewards: { ...windyEnvConfig.rewards },
    }
    return config
  }

  if (selectedEnvironment.value === 'frozenlake') {
    const config: FrozenLakeConfig = {
      environment_id: 'frozenlake',
      rows: frozenEnvConfig.rows,
      cols: frozenEnvConfig.cols,
      start: { ...frozenEnvConfig.start },
      goal: { ...frozenEnvConfig.goal },
      holes: parseCellList(holeText.value),
      slip_probability: frozenEnvConfig.slip_probability,
      rewards: { ...frozenEnvConfig.rewards },
    }
    return config
  }

  const config: GridWorldConfig = {
    environment_id: 'gridworld',
    rows: gridEnvConfig.size,
    cols: gridEnvConfig.size,
    size: gridEnvConfig.size,
    start: { ...gridEnvConfig.start },
    goal: { ...gridEnvConfig.goal },
    obstacles: parseCellList(obstacleText.value),
    traps: parseCellList(trapText.value),
    rewards: { ...gridEnvConfig.rewards },
  }
  return config
}

function buildPayload(): ExperimentRequest {
  const sharedPayload = {
    name: experimentName.value,
    submitted_by: submittedBy.value.trim() || '匿名学生',
    submission_role: viewerRole.value,
    assignment_id: selectedAssignment.value?.id ?? null,
    assignment_title: selectedAssignment.value?.title ?? null,
    environment_id: selectedEnvironment.value,
    persist_result: persistResult.value,
    env_config: buildEnvironmentConfig(),
    training: { ...training },
  }

  if (selectedAlgorithm.value === 'q_learning') {
    const request: QLearningExperimentRequest = {
      ...sharedPayload,
      algorithm_id: 'q_learning',
      algorithm_config: { ...qLearningConfig },
    }
    return request
  }

  if (selectedAlgorithm.value === 'sarsa') {
    const request: SARSAExperimentRequest = {
      ...sharedPayload,
      algorithm_id: 'sarsa',
      algorithm_config: { ...sarsaConfig },
    }
    return request
  }

  if (selectedAlgorithm.value === 'dqn') {
    const request: DQNExperimentRequest = {
      ...sharedPayload,
      algorithm_id: 'dqn',
      algorithm_config: { ...dqnConfig },
    }
    return request
  }

  const request: ReinforceExperimentRequest = {
    ...sharedPayload,
    algorithm_id: 'reinforce',
    algorithm_config: { ...reinforceConfig },
  }
  return request
}

async function handleRunExperiment() {
  errorMessage.value = ''
  streamNotice.value = ''
  loading.value = true
  currentResult.value = null
  liveRunId.value = ''
  liveMetrics.value = []
  liveLatestTrace.value = null
  liveProgress.value = 0
  liveCompletedEpisodes.value = 0
  liveTotalEpisodes.value = 0
  liveAverageReward.value = 0
  liveSuccessRate.value = 0
  streamState.value = 'running'

  try {
    streamSession = runExperimentStream(buildPayload(), {
      onStarted: (event: ExperimentStartedEvent) => {
        liveRunId.value = event.run_id
        streamState.value = 'running'
      },
      onMetric: (event) => {
        liveMetrics.value.push(event.metric)
        if (event.latest_trace) {
          liveLatestTrace.value = event.latest_trace
        }
        liveProgress.value = event.progress
        liveCompletedEpisodes.value = event.completed_episodes
        liveTotalEpisodes.value = event.total_episodes
        liveAverageReward.value = event.running_average_reward
        liveSuccessRate.value = event.running_success_rate
      },
      onControl: (event: ExperimentControlEvent) => {
        streamState.value = event.state
      },
      onCancelled: (event) => {
        streamNotice.value = event.message
        streamState.value = 'idle'
      },
    })

    const result = await streamSession.result
    if (result) {
      currentResult.value = result
      streamNotice.value = '训练已完成。'
      await loadHistory()
      if (viewerRole.value === 'teacher') {
        await loadAnalytics()
      }
    } else {
      streamNotice.value = streamNotice.value || '训练已取消。'
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : '运行实验失败。'
  } finally {
    loading.value = false
    streamState.value = 'idle'
    streamSession = null
  }
}

function formatPath() {
  if (!latestTrace.value) {
    return '当前没有采样路径数据。'
  }

  return latestTrace.value.path.map((cell) => `(${cell.row}, ${cell.col})`).join(' -> ')
}

function pauseTraining() {
  streamSession?.pause()
}

function resumeTraining() {
  streamSession?.resume()
}

function cancelTraining() {
  streamSession?.cancel()
}
</script>

<template>
  <div class="shell">
    <header class="hero">
      <div class="hero__copy">
        <p class="hero__eyebrow">毕业设计原型系统</p>
        <h1>低代码强化学习教学实验平台</h1>
        <p class="hero__summary">
          通过结构化配置快速搭建课堂实验，运行 Q-Learning、SARSA、DQN 或 REINFORCE 训练，并观察智能体在学习过程中形成的策略行为。
        </p>
      </div>

      <div class="hero__rail">
        <div v-for="item in heroMetrics" :key="item.label" class="hero__stat">
          <span>{{ item.label }}</span>
          <strong>{{ item.value }}</strong>
        </div>
      </div>
    </header>

    <main class="workspace">
      <section class="panel panel--config">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">实验设计区</p>
            <h2>配置一个教学实验</h2>
          </div>
          <button type="button" class="ghost-button" @click="loadDemoPreset">加载演示预设</button>
        </div>

        <div class="form-grid">
          <div class="section-title">工作区</div>

          <label class="field">
            <span>当前角色</span>
            <select v-model="viewerRole" class="field__select">
              <option value="student">学生</option>
              <option value="teacher">教师</option>
            </select>
          </label>

          <label class="field">
            <span>操作人名称</span>
            <input v-model="submittedBy" type="text" maxlength="40" />
          </label>

          <label class="field field--full">
            <span>实验名称</span>
            <input v-model="experimentName" type="text" maxlength="80" />
          </label>

          <div class="section-title">教学作业</div>

          <label class="field field--full">
            <span>作业模板</span>
            <select v-model="selectedAssignmentId" class="field__select">
              <option value="">自主实验</option>
              <option v-for="assignment in assignmentOptions" :key="assignment.id" :value="assignment.id">
                {{ assignment.title }}
              </option>
            </select>
          </label>

          <div v-if="selectedAssignment" class="assignment-callout field--full">
            <div class="benchmark-callout__header">
              <div>
                <p class="panel__eyebrow">教学作业</p>
                <h3>{{ selectedAssignment.title }}</h3>
              </div>
              <button type="button" class="ghost-button" @click="applySelectedAssignment">应用作业模板</button>
            </div>
            <p>{{ selectedAssignment.summary }}</p>
            <p class="benchmark-callout__note">{{ selectedAssignment.instructions }}</p>
            <div class="assignment-goals">
              <span v-for="goal in selectedAssignment.learning_goals" :key="goal">{{ goal }}</span>
            </div>
          </div>

          <div class="section-title">教师基准</div>

          <label class="field field--full">
            <span>基准模板</span>
            <select v-model="selectedBenchmarkId" class="field__select">
              <option value="">未选择教师基准</option>
              <option v-for="benchmark in benchmarkOptions" :key="benchmark.id" :value="benchmark.id">
                {{ benchmark.name }}
              </option>
            </select>
          </label>

          <div v-if="selectedBenchmark" class="benchmark-callout field--full">
            <div class="benchmark-callout__header">
              <div>
                <p class="panel__eyebrow">教师基准</p>
                <h3>{{ selectedBenchmark.name }}</h3>
              </div>
              <div class="benchmark-callout__actions">
                <button type="button" class="ghost-button" @click="applySelectedBenchmark">应用基准模板</button>
                <span v-if="selectedBenchmark.is_builtin === false" class="analytics-badge">自定义</span>
              </div>
            </div>
            <p>{{ selectedBenchmark.description }}</p>
            <p class="benchmark-callout__note">{{ selectedBenchmark.teacher_note }}</p>
            <div class="benchmark-thresholds">
              <span v-for="threshold in selectedBenchmark.thresholds" :key="threshold.metric_id">
                {{ threshold.label }} >= {{ formatBenchmarkMetric(threshold.metric_id, threshold.min_value) }}
              </span>
            </div>
          </div>

          <div class="benchmark-editor field--full">
            <div class="benchmark-callout__header">
              <div>
                <p class="panel__eyebrow">教师自定义</p>
                <h3>{{ selectedEditableBenchmark ? '编辑当前自定义基准' : '保存当前配置为教师基准' }}</h3>
              </div>
              <div class="benchmark-callout__actions">
                <button type="button" class="ghost-button" :disabled="benchmarkSaving" @click="saveBenchmarkEditor">
                  {{ benchmarkSaving ? '保存中...' : selectedEditableBenchmark ? '更新基准' : '保存为基准' }}
                </button>
                <button
                  v-if="selectedEditableBenchmark"
                  type="button"
                  class="ghost-button ghost-button--danger"
                  :disabled="benchmarkDeleting"
                  @click="removeSelectedBenchmark"
                >
                  {{ benchmarkDeleting ? '删除中...' : '删除基准' }}
                </button>
              </div>
            </div>
            <p class="benchmark-callout__note">基准会保存当前环境、算法、训练参数和阈值，可在教师视角下重复复用。</p>

            <div class="benchmark-editor__grid">
              <label class="field field--full">
                <span>基准名称</span>
                <input v-model="benchmarkEditor.name" type="text" maxlength="80" />
                <small class="field__hint">至少 3 个字符。</small>
              </label>

              <label class="field field--full">
                <span>基准说明</span>
                <textarea v-model="benchmarkEditor.description" rows="2" maxlength="240"></textarea>
              </label>

              <label class="field field--full">
                <span>教师备注</span>
                <textarea v-model="benchmarkEditor.teacher_note" rows="2" maxlength="240"></textarea>
              </label>

              <label class="field">
                <span>平均奖励阈值</span>
                <input v-model.number="benchmarkEditor.average_reward" type="number" step="0.1" />
              </label>

              <label class="field">
                <span>最佳奖励阈值</span>
                <input v-model.number="benchmarkEditor.best_reward" type="number" step="0.1" />
              </label>

              <label class="field">
                <span>成功率阈值</span>
                <input v-model.number="benchmarkEditor.success_rate" type="number" min="0" max="100" step="0.01" />
                <small class="field__hint">支持填写 0.6 或 60，都会按 60% 处理。</small>
              </label>

              <label class="field">
                <span>稳定窗口阈值</span>
                <input
                  v-model.number="benchmarkEditor.stable_success_rate"
                  type="number"
                  min="0"
                  max="100"
                  step="0.01"
                />
                <small class="field__hint">支持填写 0.7 或 70，都会按 70% 处理。</small>
              </label>
            </div>
          </div>

          <div class="section-title">实验环境</div>

          <label class="field">
            <span>网格规模</span>
            <input v-model.number="gridEnvConfig.size" type="number" min="4" max="12" :disabled="selectedEnvironment !== 'gridworld'" />
          </label>

          <label class="field">
            <span>环境类型</span>
            <select v-model="selectedEnvironment" class="field__select">
              <option v-for="environment in environmentOptions" :key="environment.id" :value="environment.id">
                {{ environment.name }}
              </option>
            </select>
          </label>

          <label class="field">
            <span>起点单元</span>
            <div class="field__inline">
              <input v-model.number="gridEnvConfig.start.row" type="number" min="0" :max="gridEnvConfig.size - 1" :disabled="selectedEnvironment !== 'gridworld'" />
              <input v-model.number="gridEnvConfig.start.col" type="number" min="0" :max="gridEnvConfig.size - 1" :disabled="selectedEnvironment !== 'gridworld'" />
            </div>
          </label>

          <label class="field">
            <span>目标单元</span>
            <div class="field__inline">
              <input v-model.number="gridEnvConfig.goal.row" type="number" min="0" :max="gridEnvConfig.size - 1" :disabled="selectedEnvironment !== 'gridworld'" />
              <input v-model.number="gridEnvConfig.goal.col" type="number" min="0" :max="gridEnvConfig.size - 1" :disabled="selectedEnvironment !== 'gridworld'" />
            </div>
          </label>

          <label class="field field--full">
            <span>障碍单元</span>
            <textarea v-model="obstacleText" rows="2" placeholder="1:2,2:2,4:2,4:3" :disabled="selectedEnvironment !== 'gridworld'"></textarea>
          </label>

          <label class="field field--full">
            <span>陷阱单元</span>
            <textarea v-model="trapText" rows="2" placeholder="2:4,4:1" :disabled="selectedEnvironment !== 'gridworld'"></textarea>
          </label>

          <label class="field">
            <span>步进惩罚</span>
            <input v-model.number="gridEnvConfig.rewards.step_penalty" type="number" step="0.1" :disabled="selectedEnvironment !== 'gridworld'" />
          </label>

          <label class="field">
            <span>目标奖励</span>
            <input v-model.number="gridEnvConfig.rewards.goal_reward" type="number" step="1" :disabled="selectedEnvironment !== 'gridworld'" />
          </label>

          <label class="field">
            <span>碰壁惩罚</span>
            <input v-model.number="gridEnvConfig.rewards.wall_penalty" type="number" step="0.1" :disabled="selectedEnvironment !== 'gridworld'" />
          </label>

          <label class="field">
            <span>陷阱惩罚</span>
            <input v-model.number="gridEnvConfig.rewards.trap_penalty" type="number" step="0.1" :disabled="selectedEnvironment !== 'gridworld'" />
          </label>

          <template v-if="selectedEnvironment === 'cliffwalking'">
            <label class="field">
              <span>悬崖环境行数</span>
              <input v-model.number="cliffEnvConfig.rows" type="number" min="4" max="12" />
            </label>

            <label class="field">
              <span>悬崖环境列数</span>
              <input v-model.number="cliffEnvConfig.cols" type="number" min="4" max="20" />
            </label>

            <label class="field">
              <span>悬崖环境起点</span>
              <div class="field__inline">
                <input v-model.number="cliffEnvConfig.start.row" type="number" min="0" :max="cliffEnvConfig.rows - 1" />
                <input v-model.number="cliffEnvConfig.start.col" type="number" min="0" :max="cliffEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field">
              <span>悬崖环境终点</span>
              <div class="field__inline">
                <input v-model.number="cliffEnvConfig.goal.row" type="number" min="0" :max="cliffEnvConfig.rows - 1" />
                <input v-model.number="cliffEnvConfig.goal.col" type="number" min="0" :max="cliffEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field field--full">
              <span>悬崖单元</span>
              <textarea v-model="cliffText" rows="2" placeholder="3:1,3:2,3:3,3:4,3:5,3:6,3:7,3:8,3:9,3:10"></textarea>
            </label>

            <label class="field">
              <span>悬崖步进惩罚</span>
              <input v-model.number="cliffEnvConfig.rewards.step_penalty" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>悬崖目标奖励</span>
              <input v-model.number="cliffEnvConfig.rewards.goal_reward" type="number" step="1" />
            </label>

            <label class="field">
              <span>悬崖碰壁惩罚</span>
              <input v-model.number="cliffEnvConfig.rewards.wall_penalty" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>悬崖坠落惩罚</span>
              <input v-model.number="cliffEnvConfig.rewards.cliff_penalty" type="number" step="1" />
            </label>
          </template>

          <template v-else-if="selectedEnvironment === 'frozenlake'">
            <label class="field">
              <span>网格行数</span>
              <input v-model.number="frozenEnvConfig.rows" type="number" min="4" max="12" />
            </label>

            <label class="field">
              <span>网格列数</span>
              <input v-model.number="frozenEnvConfig.cols" type="number" min="4" max="12" />
            </label>

            <label class="field">
              <span>起点位置</span>
              <div class="field__inline">
                <input v-model.number="frozenEnvConfig.start.row" type="number" min="0" :max="frozenEnvConfig.rows - 1" />
                <input v-model.number="frozenEnvConfig.start.col" type="number" min="0" :max="frozenEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field">
              <span>目标位置</span>
              <div class="field__inline">
                <input v-model.number="frozenEnvConfig.goal.row" type="number" min="0" :max="frozenEnvConfig.rows - 1" />
                <input v-model.number="frozenEnvConfig.goal.col" type="number" min="0" :max="frozenEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field field--full">
              <span>冰洞单元</span>
              <textarea v-model="holeText" rows="2" placeholder="1:1,1:3,2:3,3:0"></textarea>
            </label>

            <label class="field">
              <span>滑移概率</span>
              <input v-model.number="frozenEnvConfig.slip_probability" type="number" min="0" max="1" step="0.05" />
            </label>

            <label class="field">
              <span>步进奖励</span>
              <input v-model.number="frozenEnvConfig.rewards.step_penalty" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>目标奖励</span>
              <input v-model.number="frozenEnvConfig.rewards.goal_reward" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>碰壁奖励</span>
              <input v-model.number="frozenEnvConfig.rewards.wall_penalty" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>冰洞惩罚</span>
              <input v-model.number="frozenEnvConfig.rewards.hole_penalty" type="number" step="0.1" />
            </label>
          </template>

          <template v-else-if="selectedEnvironment === 'windygridworld'">
            <label class="field">
              <span>风力环境行数</span>
              <input v-model.number="windyEnvConfig.rows" type="number" min="4" max="12" />
            </label>

            <label class="field">
              <span>风力环境列数</span>
              <input v-model.number="windyEnvConfig.cols" type="number" min="5" max="20" />
            </label>

            <label class="field">
              <span>风力环境起点</span>
              <div class="field__inline">
                <input v-model.number="windyEnvConfig.start.row" type="number" min="0" :max="windyEnvConfig.rows - 1" />
                <input v-model.number="windyEnvConfig.start.col" type="number" min="0" :max="windyEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field">
              <span>风力环境终点</span>
              <div class="field__inline">
                <input v-model.number="windyEnvConfig.goal.row" type="number" min="0" :max="windyEnvConfig.rows - 1" />
                <input v-model.number="windyEnvConfig.goal.col" type="number" min="0" :max="windyEnvConfig.cols - 1" />
              </div>
            </label>

            <label class="field field--full">
              <span>列风强度</span>
              <textarea v-model="windStrengthText" rows="2" placeholder="0,0,0,1,1,1,2,2,1,0"></textarea>
            </label>

            <label class="field">
              <span>风力步进惩罚</span>
              <input v-model.number="windyEnvConfig.rewards.step_penalty" type="number" step="0.1" />
            </label>

            <label class="field">
              <span>风力目标奖励</span>
              <input v-model.number="windyEnvConfig.rewards.goal_reward" type="number" step="1" />
            </label>

            <label class="field">
              <span>风力碰壁惩罚</span>
              <input v-model.number="windyEnvConfig.rewards.wall_penalty" type="number" step="0.1" />
            </label>
          </template>

          <div class="section-title">训练算法</div>

          <label class="field field--full">
            <span>算法类型</span>
            <select v-model="selectedAlgorithm" class="field__select">
              <option v-for="algorithm in algorithmOptions" :key="algorithm.id" :value="algorithm.id">
                {{ algorithm.name }}
              </option>
            </select>
          </label>

          <template v-if="selectedAlgorithm === 'q_learning'">
            <label class="field">
              <span>学习率</span>
              <input v-model.number="qLearningConfig.learning_rate" type="number" min="0.01" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>折扣因子</span>
              <input v-model.number="qLearningConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>初始探索率</span>
              <input v-model.number="qLearningConfig.epsilon_start" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>最小探索率</span>
              <input v-model.number="qLearningConfig.epsilon_min" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>探索率衰减</span>
              <input v-model.number="qLearningConfig.epsilon_decay" type="number" min="0.8" max="1" step="0.001" />
            </label>

            <label class="field">
              <span>每轮最大步数</span>
              <input v-model.number="qLearningConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>
          </template>

          <template v-else-if="selectedAlgorithm === 'sarsa'">
            <label class="field">
              <span>学习率</span>
              <input v-model.number="sarsaConfig.learning_rate" type="number" min="0.01" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>折扣因子</span>
              <input v-model.number="sarsaConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>初始探索率</span>
              <input v-model.number="sarsaConfig.epsilon_start" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>最小探索率</span>
              <input v-model.number="sarsaConfig.epsilon_min" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>探索率衰减</span>
              <input v-model.number="sarsaConfig.epsilon_decay" type="number" min="0.8" max="1" step="0.001" />
            </label>

            <label class="field">
              <span>每轮最大步数</span>
              <input v-model.number="sarsaConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>
          </template>

          <template v-else-if="selectedAlgorithm === 'dqn'">
            <label class="field">
              <span>学习率</span>
              <input v-model.number="dqnConfig.learning_rate" type="number" min="0.0001" max="0.1" step="0.0001" />
            </label>

            <label class="field">
              <span>折扣因子</span>
              <input v-model.number="dqnConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>初始探索率</span>
              <input v-model.number="dqnConfig.epsilon_start" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>最小探索率</span>
              <input v-model.number="dqnConfig.epsilon_min" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>探索率衰减</span>
              <input v-model.number="dqnConfig.epsilon_decay" type="number" min="0.8" max="1" step="0.001" />
            </label>

            <label class="field">
              <span>每轮最大步数</span>
              <input v-model.number="dqnConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>

            <label class="field">
              <span>批大小</span>
              <input v-model.number="dqnConfig.batch_size" type="number" min="8" max="256" />
            </label>

            <label class="field">
              <span>经验回放池</span>
              <input v-model.number="dqnConfig.replay_buffer_size" type="number" min="200" max="50000" step="100" />
            </label>

            <label class="field">
              <span>目标网络同步间隔</span>
              <input v-model.number="dqnConfig.target_sync_interval" type="number" min="1" max="1000" />
            </label>

            <label class="field">
              <span>预热步数</span>
              <input v-model.number="dqnConfig.warmup_steps" type="number" min="1" max="5000" />
            </label>

            <label class="field">
              <span>隐藏层维度</span>
              <input v-model.number="dqnConfig.hidden_dim" type="number" min="16" max="512" />
            </label>
          </template>

          <template v-else>
            <label class="field">
              <span>学习率</span>
              <input v-model.number="reinforceConfig.learning_rate" type="number" min="0.001" max="0.1" step="0.001" />
            </label>

            <label class="field">
              <span>折扣因子</span>
              <input v-model.number="reinforceConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>每轮最大步数</span>
              <input v-model.number="reinforceConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>

            <label class="field">
              <span>隐藏层维度</span>
              <input v-model.number="reinforceConfig.hidden_dim" type="number" min="16" max="512" />
            </label>
          </template>

          <div class="section-title">训练配置</div>

          <label class="field">
            <span>训练轮次</span>
            <input v-model.number="training.episodes" type="number" min="20" max="2000" />
          </label>

          <label class="field">
            <span>轨迹采样频率</span>
            <input v-model.number="training.trace_frequency" type="number" min="1" max="500" />
          </label>

          <label class="field">
            <span>随机种子</span>
            <input v-model.number="training.seed" type="number" min="0" max="99999" />
          </label>
        </div>

        <div class="config-footer">
          <label class="checkbox">
            <input v-model="persistResult" type="checkbox" />
            <span>将本次实验保存到本地历史记录</span>
          </label>

          <div class="config-actions">
            <button type="button" class="ghost-button" :disabled="!canPauseStream" @click="pauseTraining">
              暂停
            </button>
            <button type="button" class="ghost-button" :disabled="!canResumeStream" @click="resumeTraining">
              继续
            </button>
            <button type="button" class="ghost-button ghost-button--danger" :disabled="!canCancelStream" @click="cancelTraining">
              取消
            </button>
            <button type="button" class="primary-button" :disabled="loading" @click="handleRunExperiment">
              {{ loading ? '训练中...' : '运行实验' }}
            </button>
          </div>
        </div>

        <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>
      </section>

      <section class="panel panel--result">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">结果面板</p>
            <h2>查看收敛过程与学习策略</h2>
          </div>
          <div class="panel__heading-actions">
            <button type="button" class="ghost-button" :disabled="!canExportReport" @click="exportExperimentReport">
              {{ reportExporting ? '导出中...' : '导出报告' }}
            </button>
            <div v-if="activeRunId" class="run-tag">{{ activeRunId }}</div>
          </div>
        </div>

        <template v-if="showResults">
          <div class="summary-grid">
            <article v-for="item in summaryCards" :key="item.label" class="summary-card">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </article>
          </div>

          <div v-if="loading" class="stream-banner">
            <div class="stream-banner__row">
              <strong>流式训练进度 | {{ streamStateLabel }}</strong>
              <span>{{ Math.round(liveProgress * 100) }}%</span>
            </div>
            <div class="stream-banner__track">
              <div class="stream-banner__fill" :style="{ width: `${Math.max(4, liveProgress * 100)}%` }"></div>
            </div>
          </div>
          <p v-else-if="streamNotice" class="analysis-note">{{ streamNotice }}</p>

          <section v-if="selectedBenchmark" class="analysis-card analysis-card--benchmark">
            <div class="analysis-card__header">
              <div>
                <p class="panel__eyebrow">教师评估</p>
                <h3>{{ selectedBenchmark.name }}</h3>
              </div>
              <span
                v-if="benchmarkEvaluation"
                class="benchmark-badge"
                :class="benchmarkEvaluation.passed ? 'benchmark-badge--pass' : 'benchmark-badge--fail'"
              >
                {{ benchmarkEvaluation.passed ? '通过' : '需改进' }}
              </span>
            </div>
            <p class="analysis-note">{{ selectedBenchmark.teacher_note }}</p>

            <template v-if="benchmarkEvaluation && benchmarkOverview">
              <div
                class="benchmark-overview"
                :class="benchmarkOverview.passed ? 'benchmark-overview--pass' : 'benchmark-overview--fail'"
              >
                <div class="benchmark-overview__copy">
                  <p class="panel__eyebrow">评估结论</p>
                  <strong class="benchmark-overview__headline">{{ benchmarkOverview.headline }}</strong>
                  <p class="benchmark-overview__message">{{ benchmarkOverview.message }}</p>
                </div>
                <div class="benchmark-overview__stats">
                  <article
                    v-for="stat in benchmarkOverviewStats"
                    :key="stat.label"
                    class="benchmark-overview__stat"
                    :class="`benchmark-overview__stat--${stat.tone}`"
                  >
                    <span>{{ stat.label }}</span>
                    <strong>{{ stat.value }}</strong>
                  </article>
                </div>
              </div>

              <div class="benchmark-section">
                <div class="benchmark-section__label">核心指标</div>
                <div class="benchmark-metric-grid">
                  <article
                    v-for="card in benchmarkMetricCards"
                    :key="card.label"
                    class="benchmark-metric-card"
                    :class="card.passed ? 'benchmark-metric-card--pass' : 'benchmark-metric-card--fail'"
                  >
                    <div class="benchmark-metric-card__row">
                      <span>{{ card.label }}</span>
                      <span
                        class="benchmark-inline-status"
                        :class="card.passed ? 'benchmark-inline-status--pass' : 'benchmark-inline-status--fail'"
                      >
                        {{ card.passed ? '达标' : '未达标' }}
                      </span>
                    </div>
                    <strong>{{ card.actual }}</strong>
                    <small>目标 {{ card.expected }}</small>
                    <p class="benchmark-metric-card__delta">{{ card.deltaText }}</p>
                    <p class="benchmark-metric-card__hint">{{ card.helpText }}</p>
                  </article>
                </div>
              </div>

              <div class="benchmark-section">
                <div class="benchmark-section__label">前置条件</div>
                <div class="benchmark-grid benchmark-grid--compact">
                  <article
                    v-for="check in benchmarkEvaluation.compatibilityChecks"
                    :key="check.label"
                    class="benchmark-check"
                    :class="check.passed ? 'benchmark-check--pass' : 'benchmark-check--fail'"
                  >
                    <div class="benchmark-check__row">
                      <strong>{{ check.label }}</strong>
                      <span>{{ check.passed ? '通过' : '未通过' }}</span>
                    </div>
                    <small>期望 {{ check.expected }} | 实际 {{ check.actual }}</small>
                    <p>{{ check.helpText }}</p>
                  </article>
                </div>
              </div>
            </template>
            <div v-else class="comparison-empty benchmark-empty">
              请先完整运行一次实验，再与所选教师基准进行评估。
            </div>
          </section>

          <div class="charts-grid">
            <MetricChart title="单轮奖励" :series="rewardSeries" accent="#0f5bd8" />
            <MetricChart title="探索率变化" :series="epsilonSeries" accent="#d97706" />
            <MetricChart title="训练误差" :series="tdErrorSeries" accent="#0f766e" />
          </div>

          <div class="analysis-grid" :class="{ 'analysis-grid--stack': widePolicyGrid }">
            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">策略视图</p>
                  <h3>学习得到的贪心策略</h3>
                </div>
              </div>
              <PolicyGrid
                v-if="finalPolicyGrid.length"
                :grid="finalPolicyGrid"
                :compact="widePolicyGrid"
                :environment-config="currentResult?.request.env_config ?? null"
              />
              <div v-else class="analysis-note">策略网格会在流式训练完成后生成。</div>
            </section>

            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">轨迹样本</p>
                  <h3>最新采样路径</h3>
                </div>
              </div>
              <div class="trace-card">
                <p v-if="latestTrace">
                  第 {{ latestTrace.episode }} 轮 |
                  {{ latestTrace.success ? '到达目标' : '未到达目标' }} |
                  奖励 {{ latestTrace.total_reward }}
                </p>
                <div v-if="latestTrace?.path.length" class="trace-step-grid" :title="formatPath()">
                  <article
                    v-for="(cell, index) in latestTrace.path"
                    :key="`latest-${latestTrace.episode}-${index}`"
                    class="trace-step-card"
                    :class="{ 'trace-step-card--terminal': index === latestTrace.path.length - 1 }"
                  >
                    <span class="trace-step-card__index">#{{ index + 1 }}</span>
                    <strong>{{ formatGridPosition(cell) }}</strong>
                  </article>
                </div>
                <p v-else class="trace-card__empty">当前没有采样路径数据。</p>
              </div>
            </section>
          </div>
        </template>

        <template v-else>
          <div class="placeholder">
            <h3>当前还没有执行任何实验。</h3>
            <p>请在左侧配置区提交 Q-Learning、SARSA、DQN 或 REINFORCE 实验，随后即可实时观察训练曲线变化。</p>
          </div>
        </template>
      </section>
    </main>

    <section class="panel panel--support">
      <div class="support-grid">
        <article class="support-card">
          <p class="panel__eyebrow">系统模块</p>
          <h3>当前后端能力范围</h3>
          <ul>
            <li>可配置 GridWorld、CliffWalking、FrozenLake 与 WindyGridWorld 实验环境</li>
            <li>Q-Learning 与 SARSA 训练服务</li>
            <li>DQN 与 REINFORCE 训练服务</li>
            <li>HTTP 与 WebSocket 实验接口</li>
          </ul>
        </article>

        <article class="support-card">
          <p class="panel__eyebrow">后续扩展</p>
          <h3>推荐的下一轮论文迭代方向</h3>
          <ul>
            <li>在当前视角切换基础上加入角色权限控制</li>
            <li>补充完整实验回放控制能力</li>
            <li>支持多作业层面的班级学习分析</li>
            <li>增加教学评价量表与成绩导出功能</li>
          </ul>
        </article>

        <article class="support-card">
          <p class="panel__eyebrow">最近实验</p>
          <h3>已保存的实验历史</h3>
          <div v-if="history.length" class="history-list">
            <div v-for="item in history" :key="item.run_id" class="history-item">
              <label class="history-toggle">
                <input
                  type="checkbox"
                  :checked="isComparisonSelected(item.run_id)"
                  @change="toggleComparison(item.run_id)"
                />
                <strong>{{ item.name }}</strong>
              </label>
              <span v-if="item.assignment_title">{{ item.assignment_title }}</span>
              <span>{{ item.submitted_by }} | {{ roleLabel(item.submission_role) }}</span>
              <span>{{ algorithmLabel(item.algorithm_id) }} | 平均奖励 {{ item.average_reward.toFixed(2) }}</span>
              <span>成功率 {{ (item.success_rate * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <p v-else class="history-empty">当前还没有保存的实验记录。</p>
        </article>
      </div>

      <section v-if="showTeacherAnalytics" class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">教师分析</p>
            <h2>查看课堂实验情况与基准达成进度</h2>
          </div>
          <div class="comparison-meta">
            <span>{{ analytics?.distinct_submitters ?? 0 }} 名提交人</span>
          </div>
        </div>

        <template v-if="analytics">
          <div class="comparison-summary">
            <article v-for="item in teacherAnalyticsCards" :key="item.label" class="summary-card">
              <span>{{ item.label }}</span>
              <strong>{{ item.value }}</strong>
            </article>
          </div>

          <div v-if="analytics.benchmark" class="analysis-note analytics-note">
            基准范围：{{ analytics.benchmark.benchmark_name }} |
            {{ analytics.benchmark.pass_count }}/{{ analytics.benchmark.evaluated_runs }} 条符合条件的学生实验通过基准。
          </div>

          <div class="analytics-grid">
            <article class="support-card">
              <p class="panel__eyebrow">作业分布</p>
              <h3>按教学任务统计的实验记录</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.assignments" :key="item.assignment_id" class="analytics-item">
                  <div>
                    <strong>{{ item.assignment_title }}</strong>
                    <p>
                      {{ item.run_count }} 次实验 | {{ item.student_runs }} 次学生实验 | 平均成功率
                      {{ (item.average_success_rate * 100).toFixed(1) }}%
                    </p>
                    <p>{{ item.distinct_submitters }} 名提交人 | 平均奖励 {{ item.average_reward.toFixed(2) }}</p>
                  </div>
                  <span v-if="item.benchmark_pass_rate !== null && item.benchmark_pass_rate !== undefined" class="analytics-badge">
                    通过率 {{ (item.benchmark_pass_rate * 100).toFixed(1) }}%
                  </span>
                </div>
              </div>
            </article>

            <article class="support-card">
              <p class="panel__eyebrow">环境分布</p>
              <h3>按训练环境统计的实验记录</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.environments" :key="item.environment_id" class="analytics-item">
                  <div>
                    <strong>{{ environmentLabel(item.environment_id) }}</strong>
                    <p>
                      {{ item.run_count }} 次实验 | {{ item.student_runs }} 次学生提交 | {{ item.distinct_submitters }} 名提交人
                    </p>
                    <p>
                      平均奖励 {{ item.average_reward.toFixed(2) }} | 平均成功率
                      {{ (item.average_success_rate * 100).toFixed(1) }}%
                    </p>
                  </div>
                  <span class="analytics-badge">
                    最佳成功率 {{ (item.best_success_rate * 100).toFixed(1) }}%
                  </span>
                </div>
              </div>
            </article>

            <article class="support-card">
              <p class="panel__eyebrow">算法分布</p>
              <h3>按算法统计的实验记录</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.algorithms" :key="item.algorithm_id" class="analytics-item">
                  <div>
                    <strong>{{ algorithmLabel(item.algorithm_id) }}</strong>
                    <p>
                      {{ item.run_count }} 次实验 | 平均奖励 {{ item.average_reward.toFixed(2) }} | 平均成功率
                      {{ (item.average_success_rate * 100).toFixed(1) }}%
                    </p>
                  </div>
                </div>
              </div>
            </article>

            <article class="support-card">
              <p class="panel__eyebrow">学生活跃情况</p>
              <h3>最近学习者概览</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.students" :key="item.submitted_by" class="analytics-item">
                  <div>
                    <strong>{{ item.submitted_by }}</strong>
                    <p>
                      {{ item.run_count }} 次实验 | 平均奖励 {{ item.average_reward.toFixed(2) }} | 最佳成功率
                      {{ (item.best_success_rate * 100).toFixed(1) }}%
                    </p>
                    <p>最近活动时间 {{ formatTimestamp(item.latest_created_at) }}</p>
                  </div>
                  <span v-if="item.benchmark_pass_count !== null && item.benchmark_pass_count !== undefined" class="analytics-badge">
                    基准通过 {{ item.benchmark_pass_count }}
                  </span>
                </div>
              </div>
            </article>
          </div>
        </template>

        <template v-else-if="analyticsLoading">
          <div class="comparison-empty">
            <h3>正在加载课堂分析数据。</h3>
            <p>教师面板正在读取已保存的实验历史和基准评估结果。</p>
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>当前没有可用的课堂分析数据。</h3>
            <p>请先保存几条学生实验记录，再切换到教师视角查看聚合结果。</p>
          </div>
        </template>
      </section>

      <section class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">轨迹回放</p>
            <h2>逐步回放采样轨迹</h2>
          </div>
          <div class="comparison-meta">
            <span>已加载 {{ replayRuns.length }} 条</span>
          </div>
        </div>

        <template v-if="replayRuns.length && replayActiveTrace">
          <div class="replay-toolbar">
            <label class="field">
              <span>回放来源</span>
              <select v-model="replaySourceRunId" class="field__select">
                <option v-for="result in replayRuns" :key="result.run_id" :value="result.run_id">
                  {{ formatReplayRunLabel(result) }}
                </option>
              </select>
            </label>

            <label class="field">
              <span>采样轮次</span>
              <select v-model.number="replayTraceEpisode" class="field__select">
                <option v-for="trace in replayTraceOptions" :key="trace.episode" :value="trace.episode">
                  第 {{ trace.episode }} 轮 | {{ trace.success ? '到达目标' : '未到达目标' }}
                </option>
              </select>
            </label>
          </div>

          <div class="replay-controls">
            <button type="button" class="ghost-button" :disabled="!canStepReplayBackward" @click="stepReplay(-1)">
              后退一步
            </button>
            <button type="button" class="ghost-button" :disabled="replayPathLength <= 1" @click="toggleReplay">
              {{ replayPlaying ? '暂停回放' : '开始回放' }}
            </button>
            <button type="button" class="ghost-button" :disabled="!canStepReplayForward" @click="stepReplay(1)">
              前进一步
            </button>
            <button type="button" class="ghost-button" :disabled="!replayPathLength" @click="resetReplay">
              重新开始
            </button>
          </div>

          <div class="replay-meta">
            <span>{{ replayProgressLabel }}</span>
            <span v-if="replayCurrentCell">
              当前单元 ({{ replayCurrentCell.row }}, {{ replayCurrentCell.col }})
            </span>
            <span>
              第 {{ replayActiveTrace.episode }} 轮 | {{ replayActiveTrace.success ? '到达目标' : '未到达目标' }}
            </span>
          </div>

          <div class="analysis-grid" :class="{ 'analysis-grid--stack': wideReplayGrid }">
            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">回放网格</p>
                  <h3>动态采样轨迹</h3>
                </div>
              </div>
              <TraceReplayGrid
                :grid="replayResultGrid"
                :trace="replayActiveTrace.path"
                :current-step="replayStepIndex"
                :compact="wideReplayGrid"
                :environment-config="replaySourceResult?.request.env_config ?? null"
              />
            </section>

            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">回放详情</p>
                  <h3>路径与奖励信息</h3>
                </div>
              </div>
              <div class="trace-card">
                <p>
                  总奖励 {{ replayActiveTrace.total_reward }} |
                  {{ replayActiveTrace.success ? '终点成功' : '终点失败' }}
                </p>
                <div v-if="replayActiveTrace.path.length" class="trace-step-grid" :title="formatReplayPath()">
                  <article
                    v-for="(cell, index) in replayActiveTrace.path"
                    :key="`replay-${replayActiveTrace.episode}-${index}`"
                    class="trace-step-card"
                    :class="{
                      'trace-step-card--current': index === replayStepIndex,
                      'trace-step-card--terminal': index === replayActiveTrace.path.length - 1,
                    }"
                  >
                    <span class="trace-step-card__index">#{{ index + 1 }}</span>
                    <strong>{{ formatGridPosition(cell) }}</strong>
                  </article>
                </div>
                <p v-else class="trace-card__empty">当前没有可供回放的采样路径。</p>
              </div>
            </section>
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>当前没有可用的回放来源。</h3>
            <p>请先运行实验，或从历史记录中选择已保存实验以开启轨迹回放。</p>
          </div>
        </template>
      </section>

      <section class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">实验对比</p>
            <h2>对比历史实验与当前结果</h2>
          </div>
          <div class="comparison-meta">
            <span>已加载 {{ compareRuns.length }} 条</span>
          </div>
        </div>

        <template v-if="comparisonReady">
          <div class="comparison-summary">
            <article v-for="row in compareSummaryRows" :key="row.runId" class="summary-card">
              <span>{{ row.label }}</span>
              <strong>{{ row.algorithm }}</strong>
              <small>
                平均 {{ row.averageReward }} | 最佳 {{ row.bestReward }} | 成功率 {{ row.successRate }} |
                {{ row.episodes }} 轮
              </small>
            </article>
          </div>

          <div class="comparison-grid">
            <ComparisonChart title="奖励趋势对比" :series="compareRewardSeries" />
            <ComparisonChart title="单轮成功情况对比" :series="compareSuccessSeries" />
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>至少需要两条实验记录才能进行对比。</h3>
            <p>
              请从历史记录中选择一条或多条已保存实验；如果当前结果可见，它会被自动加入对比。
            </p>
          </div>
        </template>
      </section>

      <div v-if="catalog" class="catalog-strip">
        <article v-for="module in [...catalog.environments, ...catalog.algorithms]" :key="module.id" class="catalog-card">
          <span class="catalog-card__label">{{ module.name }}</span>
          <p>{{ module.description }}</p>
        </article>
      </div>
    </section>
  </div>
</template>
