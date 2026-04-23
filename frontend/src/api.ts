import type {
  AssignmentCatalogResponse,
  BenchmarkCatalogResponse,
  CatalogResponse,
  ClassroomAnalyticsResponse,
  ExperimentCancelledEvent,
  ExperimentCompletedEvent,
  ExperimentControlEvent,
  ExperimentReportRequest,
  ExperimentMetricEvent,
  ExperimentRequest,
  ExperimentResult,
  ExperimentStartedEvent,
  ExperimentStreamEvent,
  HistoryEntry,
} from './types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'
const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws')

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const payload = await response.json()
      if (typeof payload.detail === 'string') {
        message = payload.detail
      }
    } catch {
      // Ignore JSON parsing failures and keep the default error message.
    }
    throw new Error(message)
  }

  return response.json() as Promise<T>
}

export function getCatalog(): Promise<CatalogResponse> {
  return request<CatalogResponse>('/api/catalog')
}

export function getBenchmarks(): Promise<BenchmarkCatalogResponse> {
  return request<BenchmarkCatalogResponse>('/api/benchmarks')
}

export function getAssignments(): Promise<AssignmentCatalogResponse> {
  return request<AssignmentCatalogResponse>('/api/assignments')
}

export function getClassroomAnalytics(
  benchmarkId?: string | null,
  assignmentId?: string | null,
): Promise<ClassroomAnalyticsResponse> {
  const search = new URLSearchParams({ limit: '60' })
  if (benchmarkId) {
    search.set('benchmark_id', benchmarkId)
  }
  if (assignmentId) {
    search.set('assignment_id', assignmentId)
  }
  return request<ClassroomAnalyticsResponse>(`/api/analytics/classroom?${search.toString()}`)
}

export function getHistory(): Promise<HistoryEntry[]> {
  return request<HistoryEntry[]>('/api/experiments/history')
}

export function getExperiment(runId: string): Promise<ExperimentResult> {
  return request<ExperimentResult>(`/api/experiments/${runId}`)
}

export function runExperiment(payload: ExperimentRequest): Promise<ExperimentResult> {
  return request<ExperimentResult>('/api/experiments/run', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function renderExperimentReport(payload: ExperimentReportRequest): Promise<Blob> {
  const response = await fetch(`${API_BASE_URL}/api/reports/render`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    let message = `Request failed with status ${response.status}`
    try {
      const data = await response.json()
      if (typeof data.detail === 'string') {
        message = data.detail
      }
    } catch {
      // Ignore JSON parsing failures and keep the default error message.
    }
    throw new Error(message)
  }

  return response.blob()
}

interface StreamHandlers {
  onStarted?: (event: ExperimentStartedEvent) => void
  onMetric?: (event: ExperimentMetricEvent) => void
  onControl?: (event: ExperimentControlEvent) => void
  onCancelled?: (event: ExperimentCancelledEvent) => void
}

export interface ExperimentStreamSession {
  result: Promise<ExperimentResult | null>
  pause: () => void
  resume: () => void
  cancel: () => void
  close: () => void
}

export function runExperimentStream(payload: ExperimentRequest, handlers: StreamHandlers = {}): ExperimentStreamSession {
  const socket = new WebSocket(`${WS_BASE_URL}/api/experiments/stream`)
  const queuedControls: string[] = []
  let settled = false
  let completed = false
  let payloadSent = false

  const sendControl = (action: 'pause' | 'resume' | 'cancel') => {
    const message = JSON.stringify({ type: action })
    if (socket.readyState === WebSocket.OPEN && payloadSent) {
      socket.send(message)
      return
    }
    queuedControls.push(message)
  }

  const result = new Promise<ExperimentResult | null>((resolve, reject) => {
    const fail = (message: string) => {
      if (settled) {
        return
      }
      settled = true
      reject(new Error(message))
    }

    socket.onopen = () => {
      socket.send(JSON.stringify(payload))
      payloadSent = true
      while (queuedControls.length > 0) {
        socket.send(queuedControls.shift() as string)
      }
    }

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data) as ExperimentStreamEvent
      switch (data.type) {
        case 'started':
          handlers.onStarted?.(data)
          break
        case 'metric':
          handlers.onMetric?.(data)
          break
        case 'control':
          handlers.onControl?.(data)
          break
        case 'cancelled':
          completed = true
          handlers.onCancelled?.(data)
          if (!settled) {
            settled = true
            resolve(null)
          }
          socket.close()
          break
        case 'completed':
          completed = true
          if (!settled) {
            settled = true
            resolve((data as ExperimentCompletedEvent).result)
          }
          socket.close()
          break
        case 'error':
          fail(typeof data.message === 'string' ? data.message : JSON.stringify(data.message))
          socket.close()
          break
      }
    }

    socket.onerror = () => {
      fail('WebSocket connection error')
      socket.close()
    }

    socket.onclose = () => {
      if (!completed && !settled) {
        fail('Training stream closed before completion')
      }
    }
  })

  return {
    result,
    pause: () => sendControl('pause'),
    resume: () => sendControl('resume'),
    cancel: () => sendControl('cancel'),
    close: () => socket.close(),
  }
}
