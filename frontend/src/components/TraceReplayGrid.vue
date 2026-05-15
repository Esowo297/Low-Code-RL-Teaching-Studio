<script setup lang="ts">
import { computed } from 'vue'

import type { EnvironmentConfig, GridPosition } from '../types'

const props = defineProps<{
  grid: string[][]
  trace: GridPosition[]
  currentStep: number
  environmentConfig?: EnvironmentConfig | null
  compact?: boolean
}>()

const columnCount = computed(() => Math.max(props.grid[0]?.length ?? 0, 1))

const labelMap: Record<string, string> = {
  U: '↑',
  R: '→',
  D: '↓',
  L: '←',
  START: 'S',
  GOAL: 'G',
  BLOCK: 'B',
  TRAP: 'T',
  CLIFF: 'CLF',
  HOLE: 'H',
}

const toneMap: Record<string, string> = {
  U: 'policy-grid__cell--move',
  R: 'policy-grid__cell--move',
  D: 'policy-grid__cell--move',
  L: 'policy-grid__cell--move',
  START: 'policy-grid__cell--start',
  GOAL: 'policy-grid__cell--goal',
  BLOCK: 'policy-grid__cell--block',
  TRAP: 'policy-grid__cell--trap',
  CLIFF: 'policy-grid__cell--cliff',
  HOLE: 'policy-grid__cell--hole',
}

const currentStepSafe = computed(() => {
  if (!props.trace.length) {
    return -1
  }
  return Math.min(Math.max(props.currentStep, 0), props.trace.length - 1)
})

const pathStepMap = computed(() => {
  const stepMap = new Map<string, number>()
  props.trace.forEach((cell, index) => {
    const key = `${cell.row}-${cell.col}`
    if (!stepMap.has(key)) {
      stepMap.set(key, index + 1)
    }
  })
  return stepMap
})

const visitedKeys = computed(() => {
  const keys = new Set<string>()
  const stopIndex = currentStepSafe.value
  if (stopIndex < 0) {
    return keys
  }

  props.trace.slice(0, stopIndex + 1).forEach((cell) => {
    keys.add(`${cell.row}-${cell.col}`)
  })
  return keys
})

const currentKey = computed(() => {
  if (currentStepSafe.value < 0) {
    return ''
  }
  const cell = props.trace[currentStepSafe.value]
  return `${cell.row}-${cell.col}`
})

function windStrengthAt(colIndex: number) {
  if (props.environmentConfig?.environment_id !== 'windygridworld') {
    return 0
  }
  return props.environmentConfig.wind_strengths[colIndex] ?? 0
}

const cells = computed(() =>
  props.grid.flatMap((row, rowIndex) =>
    row.map((token, colIndex) => {
      const id = `${rowIndex}-${colIndex}`
      const baseTone = toneMap[token] ?? 'policy-grid__cell--move'
      const isVisited = visitedKeys.value.has(id)
      const isCurrent = currentKey.value === id
      const traceStep = pathStepMap.value.get(id)
      const windStrength = windStrengthAt(colIndex)

      return {
        id,
        label: traceStep && isVisited ? `${traceStep}` : labelMap[token] ?? token,
        windLabel: windStrength > 0 ? `W${windStrength}` : '',
        classes: [
          'policy-grid__cell',
          baseTone,
          windStrength > 0 ? 'policy-grid__cell--windy' : '',
          isVisited ? 'policy-grid__cell--visited' : '',
          isCurrent ? 'policy-grid__cell--current' : '',
        ],
      }
    }),
  ),
)
</script>

<template>
  <div class="policy-grid-scroll">
    <div
      class="policy-grid"
      :class="{ 'policy-grid--compact': props.compact }"
      :style="{ '--grid-cols': `${columnCount}` }"
    >
      <div v-for="cell in cells" :key="cell.id" :class="cell.classes">
        <span class="policy-grid__main">{{ cell.label }}</span>
        <span v-if="cell.windLabel" class="policy-grid__wind">{{ cell.windLabel }}</span>
      </div>
    </div>
  </div>
</template>
