<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref, watch } from 'vue'

import type { ExperimentStreamSession } from './api'
import {
  getAssignments,
  getBenchmarks,
  getCatalog,
  getClassroomAnalytics,
  getExperiment,
  getHistory,
  renderExperimentReport,
  runExperimentStream,
} from './api'
import ComparisonChart from './components/ComparisonChart.vue'
import MetricChart from './components/MetricChart.vue'
import PolicyGrid from './components/PolicyGrid.vue'
import TraceReplayGrid from './components/TraceReplayGrid.vue'
import type {
  AlgorithmId,
  AssignmentCatalogResponse,
  AssignmentPreset,
  ClassroomAnalyticsResponse,
  BenchmarkCatalogResponse,
  BenchmarkPreset,
  BenchmarkThreshold,
  CatalogResponse,
  ComparisonSeries,
  DQNConfig,
  DQNExperimentRequest,
  EpisodeMetric,
  ExperimentControlEvent,
  ExperimentRequest,
  ExperimentResult,
  ExperimentStartedEvent,
  GridPosition,
  HistoryEntry,
  PathTrace,
  QLearningConfig,
  QLearningExperimentRequest,
  SubmissionRole,
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
const submittedBy = ref('Student A')
const reportExporting = ref(false)
const replaySourceRunId = ref('')
const replayTraceEpisode = ref(0)
const replayStepIndex = ref(0)
const replayPlaying = ref(false)
let streamSession: ExperimentStreamSession | null = null
let replayTimer: number | null = null

const obstacleText = ref('1:2,2:2,4:2,4:3')
const trapText = ref('2:4,4:1')
const experimentName = ref('GridWorld Q-Learning Demo')
const persistResult = ref(true)
const comparisonPalette = ['#0f5bd8', '#d97706', '#0f766e', '#b42318']

const envConfig = reactive({
  size: 6,
  start: { row: 0, col: 0 },
  goal: { row: 5, col: 5 },
  obstacles: [] as GridPosition[],
  traps: [] as GridPosition[],
  rewards: {
    step_penalty: -1,
    goal_reward: 20,
    wall_penalty: -3,
    trap_penalty: -10,
  },
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

const heroMetrics = computed(() => [
  { label: 'Focus', value: 'Low-Code RL Lab' },
  { label: 'Workspace', value: roleLabel(viewerRole.value) },
  { label: 'Operator', value: submittedBy.value || 'Unassigned' },
])

const activeMetrics = computed(() => currentResult.value?.metrics ?? liveMetrics.value)
const rewardSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.reward })))
const epsilonSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.epsilon })))
const tdErrorSeries = computed(() => activeMetrics.value.map((metric) => ({ x: metric.episode, y: metric.td_error })))

const summaryCards = computed(() => {
  if (currentResult.value) {
    return [
      { label: 'Average Reward', value: currentResult.value.summary.average_reward.toFixed(3) },
      { label: 'Best Reward', value: currentResult.value.summary.best_reward.toFixed(3) },
      { label: 'Success Rate', value: `${(currentResult.value.summary.success_rate * 100).toFixed(1)}%` },
      { label: 'Stable Window', value: `${(currentResult.value.summary.stable_success_rate * 100).toFixed(1)}%` },
    ]
  }

  if (!liveMetrics.value.length) {
    return []
  }

  const bestReward = Math.max(...liveMetrics.value.map((metric) => metric.reward))
  return [
    { label: 'Average Reward', value: liveAverageReward.value.toFixed(3) },
    { label: 'Best Reward', value: bestReward.toFixed(3) },
    { label: 'Success Rate', value: `${(liveSuccessRate.value * 100).toFixed(1)}%` },
    { label: 'Progress', value: `${liveCompletedEpisodes.value}/${liveTotalEpisodes.value} (${Math.round(liveProgress.value * 100)}%)` },
  ]
})

const latestTrace = computed(() => currentResult.value?.path_traces.at(-1) ?? liveLatestTrace.value)
const algorithmOptions = computed(() => catalog.value?.algorithms ?? [])
const assignmentOptions = computed(() => assignmentCatalog.value?.assignments ?? [])
const selectedAssignment = computed<AssignmentPreset | null>(
  () => assignmentOptions.value.find((assignment) => assignment.id === selectedAssignmentId.value) ?? null,
)
const benchmarkOptions = computed(() => benchmarkCatalog.value?.benchmarks ?? [])
const effectiveBenchmarkId = computed(() => {
  if (selectedBenchmarkId.value) {
    return selectedBenchmarkId.value
  }
  return selectedAssignment.value?.benchmark_id ?? ''
})
const selectedBenchmark = computed<BenchmarkPreset | null>(
  () => benchmarkOptions.value.find((benchmark) => benchmark.id === effectiveBenchmarkId.value) ?? null,
)
const activeRunId = computed(() => currentResult.value?.run_id ?? liveRunId.value)
const showResults = computed(() => Boolean(currentResult.value || liveMetrics.value.length || loading.value))
const showTeacherAnalytics = computed(() => viewerRole.value === 'teacher')
const finalPolicyGrid = computed(() => currentResult.value?.policy_grid ?? [])
const canExportReport = computed(() => Boolean(currentResult.value) && !loading.value && !reportExporting.value)
const canPauseStream = computed(() => loading.value && streamState.value === 'running')
const canResumeStream = computed(() => loading.value && streamState.value === 'paused')
const canCancelStream = computed(() => loading.value && streamState.value !== 'cancelling')
const streamStateLabel = computed(() => {
  if (streamState.value === 'paused') {
    return 'Paused'
  }
  if (streamState.value === 'cancelling') {
    return 'Cancelling'
  }
  return 'Running'
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
    return 'No sampled path loaded.'
  }
  return `Step ${replayStepIndex.value + 1}/${replayPathLength.value}`
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
    { label: 'Total Runs', value: `${analytics.value.total_runs}` },
    { label: 'Student Runs', value: `${analytics.value.student_runs}` },
    {
      label: analytics.value.benchmark ? 'Benchmark Pass Rate' : 'Teacher Runs',
      value: analytics.value.benchmark
        ? `${(analytics.value.benchmark.pass_rate * 100).toFixed(1)}%`
        : `${analytics.value.teacher_runs}`,
    },
    { label: 'Average Success', value: `${(analytics.value.average_success_rate * 100).toFixed(1)}%` },
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
      label: 'Algorithm Match',
      passed: result.request.algorithm_id === benchmark.request.algorithm_id,
      expected: algorithmLabel(benchmark.request.algorithm_id),
      actual: algorithmLabel(result.request.algorithm_id),
      helpText: 'Teacher baselines should be compared against the same algorithm family.',
    },
    {
      label: 'Grid Size Match',
      passed: result.request.env_config.size === benchmark.request.env_config.size,
      expected: `${benchmark.request.env_config.size} x ${benchmark.request.env_config.size}`,
      actual: `${result.request.env_config.size} x ${result.request.env_config.size}`,
      helpText: 'Changing the environment size changes the difficulty and invalidates direct comparison.',
    },
    {
      label: 'Episode Budget',
      passed: result.request.training.episodes >= benchmark.request.training.episodes,
      expected: `>= ${benchmark.request.training.episodes}`,
      actual: `${result.request.training.episodes}`,
      helpText: 'Shorter runs are not directly comparable to the teacher baseline thresholds.',
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

watch(
  () => envConfig.size,
  (size) => {
    envConfig.goal.row = size - 1
    envConfig.goal.col = size - 1
    if (size !== 6) {
      obstacleText.value = ''
      trapText.value = ''
    }
  },
)

watch(selectedAlgorithm, (algorithmId) => {
  experimentName.value = algorithmId === 'q_learning' ? 'GridWorld Q-Learning Demo' : 'GridWorld DQN Demo'
})

watch(viewerRole, (role, previousRole) => {
  const currentName = submittedBy.value.trim()
  const wasDefaultStudent = currentName === '' || currentName === 'Student A'
  const wasDefaultTeacher = currentName === 'Course Instructor'
  if ((previousRole === 'student' && wasDefaultStudent) || (previousRole === 'teacher' && wasDefaultTeacher)) {
    submittedBy.value = role === 'teacher' ? 'Course Instructor' : 'Student A'
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
  return algorithmId === 'q_learning' ? 'Q-Learning' : 'DQN'
}

function roleLabel(role: SubmissionRole) {
  return role === 'teacher' ? 'Teacher View' : 'Student View'
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
        throw new Error(`Invalid cell format: ${pair}. Use row:col pairs such as 1:2,2:2`)
      }
      return { row, col }
    })
}

function serializeCellList(cells: GridPosition[]) {
  return cells.map((cell) => `${cell.row}:${cell.col}`).join(',')
}

function comparisonLabel(result: ExperimentResult, isCurrent = false) {
  const prefix = isCurrent ? 'Current' : result.request.name
  return `${prefix} (${algorithmLabel(result.request.algorithm_id)})`
}

function formatBenchmarkMetric(metricId: BenchmarkThreshold['metric_id'], value: number) {
  if (metricId === 'success_rate' || metricId === 'stable_success_rate') {
    return `${(value * 100).toFixed(1)}%`
  }
  return value.toFixed(3)
}

function formatTimestamp(timestamp: string) {
  return new Date(timestamp).toLocaleString()
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
    errorMessage.value = 'You can compare up to three saved runs at once.'
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
    errorMessage.value = error instanceof Error ? error.message : 'Failed to load comparison run.'
  } finally {
    comparisonLoadingIds.value = comparisonLoadingIds.value.filter((id) => id !== runId)
  }
}

function loadDemoPreset() {
  selectedAssignmentId.value = ''
  selectedBenchmarkId.value = ''
  experimentName.value = selectedAlgorithm.value === 'q_learning' ? 'GridWorld Q-Learning Demo' : 'GridWorld DQN Demo'
  envConfig.size = 6
  envConfig.start = { row: 0, col: 0 }
  envConfig.goal = { row: 5, col: 5 }
  envConfig.rewards = {
    step_penalty: -1,
    goal_reward: 20,
    wall_penalty: -3,
    trap_penalty: -10,
  }
  Object.assign(qLearningConfig, {
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
  Object.assign(training, {
    episodes: 120,
    seed: 7,
    trace_frequency: 30,
  })
  obstacleText.value = '1:2,2:2,4:2,4:3'
  trapText.value = '2:4,4:1'
}

function applyExperimentRequestPreset(request: ExperimentRequest) {
  selectedAlgorithm.value = request.algorithm_id
  experimentName.value = request.name
  envConfig.size = request.env_config.size
  envConfig.start = { ...request.env_config.start }
  envConfig.goal = { ...request.env_config.goal }
  envConfig.rewards = { ...request.env_config.rewards }
  obstacleText.value = serializeCellList(request.env_config.obstacles)
  trapText.value = serializeCellList(request.env_config.traps)

  training.episodes = request.training.episodes
  training.seed = request.training.seed
  training.trace_frequency = request.training.trace_frequency

  if (request.algorithm_id === 'q_learning') {
    Object.assign(qLearningConfig, request.algorithm_config)
  } else {
    Object.assign(dqnConfig, request.algorithm_config)
  }
}

function applySelectedBenchmark() {
  if (!selectedBenchmark.value) {
    return
  }

  applyExperimentRequestPreset(selectedBenchmark.value.request)
  streamNotice.value = `Loaded teacher benchmark preset: ${selectedBenchmark.value.name}`
}

function applySelectedAssignment() {
  if (!selectedAssignment.value) {
    return
  }

  selectedBenchmarkId.value = selectedAssignment.value.benchmark_id ?? ''
  applyExperimentRequestPreset(selectedAssignment.value.request)
  experimentName.value = selectedAssignment.value.request.name
  streamNotice.value = `Loaded assignment template: ${selectedAssignment.value.title}`
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
    streamNotice.value = 'Teaching report exported as Markdown.'
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Failed to export report.'
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

function formatReplayPath() {
  if (!replayActiveTrace.value || !replayPathLength.value) {
    return 'No sampled path available for replay.'
  }

  return replayActiveTrace.value.path
    .map((cell, index) => `${index === replayStepIndex.value ? '[' : ''}(${cell.row},${cell.col})${index === replayStepIndex.value ? ']' : ''}`)
    .join(' -> ')
}

function buildPayload(): ExperimentRequest {
  const sharedPayload = {
    name: experimentName.value,
    environment_id: 'gridworld' as const,
    persist_result: persistResult.value,
    env_config: {
      ...envConfig,
      start: { ...envConfig.start },
      goal: { ...envConfig.goal },
      rewards: { ...envConfig.rewards },
      obstacles: parseCellList(obstacleText.value),
      traps: parseCellList(trapText.value),
    },
    training: { ...training },
  }

  if (selectedAlgorithm.value === 'q_learning') {
    const request: QLearningExperimentRequest = {
      ...sharedPayload,
      submitted_by: submittedBy.value.trim() || 'Anonymous Student',
      submission_role: viewerRole.value,
      assignment_id: selectedAssignment.value?.id ?? null,
      assignment_title: selectedAssignment.value?.title ?? null,
      algorithm_id: 'q_learning',
      algorithm_config: { ...qLearningConfig },
    }
    return request
  }

  const request: DQNExperimentRequest = {
    ...sharedPayload,
    submitted_by: submittedBy.value.trim() || 'Anonymous Student',
    submission_role: viewerRole.value,
    assignment_id: selectedAssignment.value?.id ?? null,
    assignment_title: selectedAssignment.value?.title ?? null,
    algorithm_id: 'dqn',
    algorithm_config: { ...dqnConfig },
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
      streamNotice.value = 'Training completed.'
      await loadHistory()
      if (viewerRole.value === 'teacher') {
        await loadAnalytics()
      }
    } else {
      streamNotice.value = streamNotice.value || 'Training cancelled.'
    }
  } catch (error) {
    errorMessage.value = error instanceof Error ? error.message : 'Failed to run experiment.'
  } finally {
    loading.value = false
    streamState.value = 'idle'
    streamSession = null
  }
}

function formatPath() {
  if (!latestTrace.value) {
    return 'No sampled path available.'
  }

  return latestTrace.value.path.map((cell) => `(${cell.row},${cell.col})`).join(' -> ')
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
        <p class="hero__eyebrow">Graduation Project Prototype</p>
        <h1>Low-Code Reinforcement Learning Teaching Studio</h1>
        <p class="hero__summary">
          Build a classroom experiment through structured configuration, execute Q-learning or DQN training, and
          inspect the policy that emerges from learning.
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
            <p class="panel__eyebrow">Experiment Designer</p>
            <h2>Configure a teaching experiment</h2>
          </div>
          <button type="button" class="ghost-button" @click="loadDemoPreset">Load demo preset</button>
        </div>

        <div class="form-grid">
          <div class="section-title">Workspace</div>

          <label class="field">
            <span>Current Role</span>
            <select v-model="viewerRole" class="field__select">
              <option value="student">Student</option>
              <option value="teacher">Teacher</option>
            </select>
          </label>

          <label class="field">
            <span>Operator Name</span>
            <input v-model="submittedBy" type="text" maxlength="40" />
          </label>

          <label class="field field--full">
            <span>Experiment Name</span>
            <input v-model="experimentName" type="text" maxlength="80" />
          </label>

          <div class="section-title">Teaching Assignment</div>

          <label class="field field--full">
            <span>Assignment Template</span>
            <select v-model="selectedAssignmentId" class="field__select">
              <option value="">Independent experiment</option>
              <option v-for="assignment in assignmentOptions" :key="assignment.id" :value="assignment.id">
                {{ assignment.title }}
              </option>
            </select>
          </label>

          <div v-if="selectedAssignment" class="assignment-callout field--full">
            <div class="benchmark-callout__header">
              <div>
                <p class="panel__eyebrow">Teaching Assignment</p>
                <h3>{{ selectedAssignment.title }}</h3>
              </div>
              <button type="button" class="ghost-button" @click="applySelectedAssignment">Apply assignment template</button>
            </div>
            <p>{{ selectedAssignment.summary }}</p>
            <p class="benchmark-callout__note">{{ selectedAssignment.instructions }}</p>
            <div class="assignment-goals">
              <span v-for="goal in selectedAssignment.learning_goals" :key="goal">{{ goal }}</span>
            </div>
          </div>

          <div class="section-title">Teacher Benchmark</div>

          <label class="field field--full">
            <span>Benchmark Preset</span>
            <select v-model="selectedBenchmarkId" class="field__select">
              <option value="">No benchmark selected</option>
              <option v-for="benchmark in benchmarkOptions" :key="benchmark.id" :value="benchmark.id">
                {{ benchmark.name }}
              </option>
            </select>
          </label>

          <div v-if="selectedBenchmark" class="benchmark-callout field--full">
            <div class="benchmark-callout__header">
              <div>
                <p class="panel__eyebrow">Teacher Benchmark</p>
                <h3>{{ selectedBenchmark.name }}</h3>
              </div>
              <button type="button" class="ghost-button" @click="applySelectedBenchmark">Apply benchmark preset</button>
            </div>
            <p>{{ selectedBenchmark.description }}</p>
            <p class="benchmark-callout__note">{{ selectedBenchmark.teacher_note }}</p>
            <div class="benchmark-thresholds">
              <span v-for="threshold in selectedBenchmark.thresholds" :key="threshold.metric_id">
                {{ threshold.label }} >= {{ formatBenchmarkMetric(threshold.metric_id, threshold.min_value) }}
              </span>
            </div>
          </div>

          <div class="section-title">Environment</div>

          <label class="field">
            <span>Grid Size</span>
            <input v-model.number="envConfig.size" type="number" min="4" max="12" />
          </label>

          <label class="field">
            <span>Goal Cell</span>
            <div class="field__inline">
              <input v-model.number="envConfig.goal.row" type="number" min="0" />
              <input v-model.number="envConfig.goal.col" type="number" min="0" />
            </div>
          </label>

          <label class="field field--full">
            <span>Obstacle Cells</span>
            <textarea v-model="obstacleText" rows="2" placeholder="1:2,2:2,4:2,4:3"></textarea>
          </label>

          <label class="field field--full">
            <span>Trap Cells</span>
            <textarea v-model="trapText" rows="2" placeholder="2:4,4:1"></textarea>
          </label>

          <label class="field">
            <span>Step Penalty</span>
            <input v-model.number="envConfig.rewards.step_penalty" type="number" step="0.1" />
          </label>

          <label class="field">
            <span>Goal Reward</span>
            <input v-model.number="envConfig.rewards.goal_reward" type="number" step="1" />
          </label>

          <label class="field">
            <span>Wall Penalty</span>
            <input v-model.number="envConfig.rewards.wall_penalty" type="number" step="0.1" />
          </label>

          <label class="field">
            <span>Trap Penalty</span>
            <input v-model.number="envConfig.rewards.trap_penalty" type="number" step="0.1" />
          </label>

          <div class="section-title">Algorithm</div>

          <label class="field field--full">
            <span>Algorithm Type</span>
            <select v-model="selectedAlgorithm" class="field__select">
              <option v-for="algorithm in algorithmOptions" :key="algorithm.id" :value="algorithm.id">
                {{ algorithm.name }}
              </option>
            </select>
          </label>

          <template v-if="selectedAlgorithm === 'q_learning'">
            <label class="field">
              <span>Learning Rate</span>
              <input v-model.number="qLearningConfig.learning_rate" type="number" min="0.01" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>Discount Factor</span>
              <input v-model.number="qLearningConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>Epsilon Start</span>
              <input v-model.number="qLearningConfig.epsilon_start" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>Epsilon Min</span>
              <input v-model.number="qLearningConfig.epsilon_min" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>Epsilon Decay</span>
              <input v-model.number="qLearningConfig.epsilon_decay" type="number" min="0.8" max="1" step="0.001" />
            </label>

            <label class="field">
              <span>Max Steps / Episode</span>
              <input v-model.number="qLearningConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>
          </template>

          <template v-else>
            <label class="field">
              <span>Learning Rate</span>
              <input v-model.number="dqnConfig.learning_rate" type="number" min="0.0001" max="0.1" step="0.0001" />
            </label>

            <label class="field">
              <span>Discount Factor</span>
              <input v-model.number="dqnConfig.gamma" type="number" min="0.5" max="0.999" step="0.001" />
            </label>

            <label class="field">
              <span>Epsilon Start</span>
              <input v-model.number="dqnConfig.epsilon_start" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>Epsilon Min</span>
              <input v-model.number="dqnConfig.epsilon_min" type="number" min="0" max="1" step="0.01" />
            </label>

            <label class="field">
              <span>Epsilon Decay</span>
              <input v-model.number="dqnConfig.epsilon_decay" type="number" min="0.8" max="1" step="0.001" />
            </label>

            <label class="field">
              <span>Max Steps / Episode</span>
              <input v-model.number="dqnConfig.max_steps_per_episode" type="number" min="10" max="400" />
            </label>

            <label class="field">
              <span>Batch Size</span>
              <input v-model.number="dqnConfig.batch_size" type="number" min="8" max="256" />
            </label>

            <label class="field">
              <span>Replay Buffer</span>
              <input v-model.number="dqnConfig.replay_buffer_size" type="number" min="200" max="50000" step="100" />
            </label>

            <label class="field">
              <span>Target Sync</span>
              <input v-model.number="dqnConfig.target_sync_interval" type="number" min="1" max="1000" />
            </label>

            <label class="field">
              <span>Warmup Steps</span>
              <input v-model.number="dqnConfig.warmup_steps" type="number" min="1" max="5000" />
            </label>

            <label class="field">
              <span>Hidden Dim</span>
              <input v-model.number="dqnConfig.hidden_dim" type="number" min="16" max="512" />
            </label>
          </template>

          <div class="section-title">Training</div>

          <label class="field">
            <span>Episodes</span>
            <input v-model.number="training.episodes" type="number" min="20" max="2000" />
          </label>

          <label class="field">
            <span>Trace Frequency</span>
            <input v-model.number="training.trace_frequency" type="number" min="1" max="500" />
          </label>

          <label class="field">
            <span>Random Seed</span>
            <input v-model.number="training.seed" type="number" min="0" max="99999" />
          </label>
        </div>

        <div class="config-footer">
          <label class="checkbox">
            <input v-model="persistResult" type="checkbox" />
            <span>Persist this run to local history</span>
          </label>

          <div class="config-actions">
            <button type="button" class="ghost-button" :disabled="!canPauseStream" @click="pauseTraining">
              Pause
            </button>
            <button type="button" class="ghost-button" :disabled="!canResumeStream" @click="resumeTraining">
              Resume
            </button>
            <button type="button" class="ghost-button ghost-button--danger" :disabled="!canCancelStream" @click="cancelTraining">
              Cancel
            </button>
            <button type="button" class="primary-button" :disabled="loading" @click="handleRunExperiment">
              {{ loading ? 'Training...' : 'Run Experiment' }}
            </button>
          </div>
        </div>

        <p v-if="errorMessage" class="error-banner">{{ errorMessage }}</p>
      </section>

      <section class="panel panel--result">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">Result Console</p>
            <h2>Inspect convergence and learned policy</h2>
          </div>
          <div class="panel__heading-actions">
            <button type="button" class="ghost-button" :disabled="!canExportReport" @click="exportExperimentReport">
              {{ reportExporting ? 'Exporting...' : 'Export Report' }}
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
              <strong>Streaming training progress | {{ streamStateLabel }}</strong>
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
                <p class="panel__eyebrow">Teacher Evaluation</p>
                <h3>{{ selectedBenchmark.name }}</h3>
              </div>
              <span
                v-if="benchmarkEvaluation"
                class="benchmark-badge"
                :class="benchmarkEvaluation.passed ? 'benchmark-badge--pass' : 'benchmark-badge--fail'"
              >
                {{ benchmarkEvaluation.passed ? 'Pass' : 'Needs Revision' }}
              </span>
            </div>
            <p class="analysis-note">{{ selectedBenchmark.teacher_note }}</p>

            <div v-if="benchmarkEvaluation" class="benchmark-grid">
              <article
                v-for="check in [...benchmarkEvaluation.compatibilityChecks, ...benchmarkEvaluation.thresholdChecks]"
                :key="check.label"
                class="benchmark-check"
                :class="check.passed ? 'benchmark-check--pass' : 'benchmark-check--fail'"
              >
                <div class="benchmark-check__row">
                  <strong>{{ check.label }}</strong>
                  <span>{{ check.passed ? 'Pass' : 'Fail' }}</span>
                </div>
                <small>Expected {{ check.expected }} | Actual {{ check.actual }}</small>
                <p>{{ check.helpText }}</p>
              </article>
            </div>
            <div v-else class="comparison-empty benchmark-empty">
              Run a complete experiment to evaluate it against the selected teacher baseline.
            </div>
          </section>

          <div class="charts-grid">
            <MetricChart title="Episode Reward" :series="rewardSeries" accent="#0f5bd8" />
            <MetricChart title="Exploration Rate" :series="epsilonSeries" accent="#d97706" />
            <MetricChart title="TD Error" :series="tdErrorSeries" accent="#0f766e" />
          </div>

          <div class="analysis-grid">
            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">Policy View</p>
                  <h3>Learned greedy policy</h3>
                </div>
              </div>
              <PolicyGrid v-if="finalPolicyGrid.length" :grid="finalPolicyGrid" />
              <div v-else class="analysis-note">Policy grid is generated after the streaming run finishes.</div>
            </section>

            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">Trace Sample</p>
                  <h3>Latest sampled path</h3>
                </div>
              </div>
              <div class="trace-card">
                <p v-if="latestTrace">
                  Episode {{ latestTrace.episode }} |
                  {{ latestTrace.success ? 'Goal reached' : 'Goal not reached' }} |
                  Reward {{ latestTrace.total_reward }}
                </p>
                <code>{{ formatPath() }}</code>
              </div>
            </section>
          </div>
        </template>

        <template v-else>
          <div class="placeholder">
            <h3>No experiment has been executed yet.</h3>
            <p>Use the left-side designer to submit a Q-learning or DQN run and watch the curves update live.</p>
          </div>
        </template>
      </section>
    </main>

    <section class="panel panel--support">
      <div class="support-grid">
        <article class="support-card">
          <p class="panel__eyebrow">System Modules</p>
          <h3>Current backend scope</h3>
          <ul>
            <li>Configurable GridWorld environment</li>
            <li>Q-learning training service</li>
            <li>DQN training service with replay memory</li>
            <li>HTTP and WebSocket experiment APIs</li>
          </ul>
        </article>

        <article class="support-card">
          <p class="panel__eyebrow">Planned Upgrades</p>
          <h3>Recommended next thesis iteration</h3>
          <ul>
            <li>Add role-based access control beyond the current view switch</li>
            <li>Add replay controls for finished runs</li>
            <li>Add cohort-level learning analytics over multiple assignments</li>
            <li>Add rubric and grade export for teaching assessment</li>
          </ul>
        </article>

        <article class="support-card">
          <p class="panel__eyebrow">Recent Runs</p>
          <h3>Saved experiment history</h3>
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
              <span>{{ algorithmLabel(item.algorithm_id) }} | avg {{ item.average_reward.toFixed(2) }}</span>
              <span>success {{ (item.success_rate * 100).toFixed(1) }}%</span>
            </div>
          </div>
          <p v-else class="history-empty">No persisted runs yet.</p>
        </article>
      </div>

      <section v-if="showTeacherAnalytics" class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">Teacher Analytics</p>
            <h2>Review classroom activity and benchmark progress</h2>
          </div>
          <div class="comparison-meta">
            <span>{{ analytics?.distinct_submitters ?? 0 }} submitters</span>
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
            Benchmark scope: {{ analytics.benchmark.benchmark_name }} |
            {{ analytics.benchmark.pass_count }}/{{ analytics.benchmark.evaluated_runs }} eligible student runs passed.
          </div>

          <div class="analytics-grid">
            <article class="support-card">
              <p class="panel__eyebrow">Assignment Distribution</p>
              <h3>Saved runs by teaching task</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.assignments" :key="item.assignment_id" class="analytics-item">
                  <div>
                    <strong>{{ item.assignment_title }}</strong>
                    <p>
                      {{ item.run_count }} runs | {{ item.student_runs }} student runs | avg success
                      {{ (item.average_success_rate * 100).toFixed(1) }}%
                    </p>
                    <p>{{ item.distinct_submitters }} submitters | avg reward {{ item.average_reward.toFixed(2) }}</p>
                  </div>
                  <span v-if="item.benchmark_pass_rate !== null && item.benchmark_pass_rate !== undefined" class="analytics-badge">
                    pass {{ (item.benchmark_pass_rate * 100).toFixed(1) }}%
                  </span>
                </div>
              </div>
            </article>

            <article class="support-card">
              <p class="panel__eyebrow">Algorithm Distribution</p>
              <h3>Saved runs by algorithm</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.algorithms" :key="item.algorithm_id" class="analytics-item">
                  <div>
                    <strong>{{ algorithmLabel(item.algorithm_id) }}</strong>
                    <p>
                      {{ item.run_count }} runs | avg reward {{ item.average_reward.toFixed(2) }} | avg success
                      {{ (item.average_success_rate * 100).toFixed(1) }}%
                    </p>
                  </div>
                </div>
              </div>
            </article>

            <article class="support-card">
              <p class="panel__eyebrow">Student Activity</p>
              <h3>Recent learner summary</h3>
              <div class="analytics-list">
                <div v-for="item in analytics.students" :key="item.submitted_by" class="analytics-item">
                  <div>
                    <strong>{{ item.submitted_by }}</strong>
                    <p>
                      {{ item.run_count }} runs | avg reward {{ item.average_reward.toFixed(2) }} | best success
                      {{ (item.best_success_rate * 100).toFixed(1) }}%
                    </p>
                    <p>Latest activity {{ formatTimestamp(item.latest_created_at) }}</p>
                  </div>
                  <span v-if="item.benchmark_pass_count !== null && item.benchmark_pass_count !== undefined" class="analytics-badge">
                    benchmark pass {{ item.benchmark_pass_count }}
                  </span>
                </div>
              </div>
            </article>
          </div>
        </template>

        <template v-else-if="analyticsLoading">
          <div class="comparison-empty">
            <h3>Loading classroom analytics.</h3>
            <p>The teacher dashboard is reading saved experiment history and benchmark outcomes.</p>
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>No classroom analytics available.</h3>
            <p>Persist a few student runs first, then switch to teacher view to inspect aggregated results.</p>
          </div>
        </template>
      </section>

      <section class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">Replay Lab</p>
            <h2>Replay sampled trajectories step by step</h2>
          </div>
          <div class="comparison-meta">
            <span>{{ replayRuns.length }} loaded</span>
          </div>
        </div>

        <template v-if="replayRuns.length && replayActiveTrace">
          <div class="replay-toolbar">
            <label class="field">
              <span>Replay Source</span>
              <select v-model="replaySourceRunId" class="field__select">
                <option v-for="result in replayRuns" :key="result.run_id" :value="result.run_id">
                  {{ formatReplayRunLabel(result) }}
                </option>
              </select>
            </label>

            <label class="field">
              <span>Sampled Episode</span>
              <select v-model.number="replayTraceEpisode" class="field__select">
                <option v-for="trace in replayTraceOptions" :key="trace.episode" :value="trace.episode">
                  Episode {{ trace.episode }} | {{ trace.success ? 'Goal reached' : 'Goal not reached' }}
                </option>
              </select>
            </label>
          </div>

          <div class="replay-controls">
            <button type="button" class="ghost-button" :disabled="!canStepReplayBackward" @click="stepReplay(-1)">
              Step Back
            </button>
            <button type="button" class="ghost-button" :disabled="replayPathLength <= 1" @click="toggleReplay">
              {{ replayPlaying ? 'Pause Replay' : 'Play Replay' }}
            </button>
            <button type="button" class="ghost-button" :disabled="!canStepReplayForward" @click="stepReplay(1)">
              Step Forward
            </button>
            <button type="button" class="ghost-button" :disabled="!replayPathLength" @click="resetReplay">
              Restart
            </button>
          </div>

          <div class="replay-meta">
            <span>{{ replayProgressLabel }}</span>
            <span v-if="replayCurrentCell">
              Current Cell ({{ replayCurrentCell.row }}, {{ replayCurrentCell.col }})
            </span>
            <span>
              Episode {{ replayActiveTrace.episode }} | {{ replayActiveTrace.success ? 'Goal reached' : 'Goal not reached' }}
            </span>
          </div>

          <div class="analysis-grid">
            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">Replay Grid</p>
                  <h3>Animated sampled trajectory</h3>
                </div>
              </div>
              <TraceReplayGrid
                :grid="replaySourceResult?.policy_grid ?? []"
                :trace="replayActiveTrace.path"
                :current-step="replayStepIndex"
              />
            </section>

            <section class="analysis-card">
              <div class="analysis-card__header">
                <div>
                  <p class="panel__eyebrow">Replay Detail</p>
                  <h3>Path and reward context</h3>
                </div>
              </div>
              <div class="trace-card">
                <p>
                  Total reward {{ replayActiveTrace.total_reward }} |
                  {{ replayActiveTrace.success ? 'Terminal success' : 'Terminal failure' }}
                </p>
                <code>{{ formatReplayPath() }}</code>
              </div>
            </section>
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>No replay source available.</h3>
            <p>Run an experiment or select saved runs from history to open trajectory replay.</p>
          </div>
        </template>
      </section>

      <section class="comparison-panel">
        <div class="panel__heading">
          <div>
            <p class="panel__eyebrow">Comparison Lab</p>
            <h2>Compare saved runs and current output</h2>
          </div>
          <div class="comparison-meta">
            <span>{{ compareRuns.length }} loaded</span>
          </div>
        </div>

        <template v-if="comparisonReady">
          <div class="comparison-summary">
            <article v-for="row in compareSummaryRows" :key="row.runId" class="summary-card">
              <span>{{ row.label }}</span>
              <strong>{{ row.algorithm }}</strong>
              <small>
                avg {{ row.averageReward }} | best {{ row.bestReward }} | success {{ row.successRate }} |
                {{ row.episodes }} eps
              </small>
            </article>
          </div>

          <div class="comparison-grid">
            <ComparisonChart title="Reward Trend Comparison" :series="compareRewardSeries" />
            <ComparisonChart title="Episode Success Comparison" :series="compareSuccessSeries" />
          </div>
        </template>

        <template v-else>
          <div class="comparison-empty">
            <h3>Comparison needs at least two runs.</h3>
            <p>
              Select one or more saved runs from history. If a current run is visible, it will be included
              automatically.
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
