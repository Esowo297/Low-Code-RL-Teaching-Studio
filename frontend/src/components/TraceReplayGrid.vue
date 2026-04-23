<script setup lang="ts">
import { computed } from 'vue'

import type { GridPosition } from '../types'

const props = defineProps<{
  grid: string[][]
  trace: GridPosition[]
  currentStep: number
}>()

const labelMap: Record<string, string> = {
  U: 'UP',
  R: 'RT',
  D: 'DN',
  L: 'LT',
  START: 'ST',
  GOAL: 'GO',
  BLOCK: 'XX',
  TRAP: 'TP',
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

const cells = computed(() =>
  props.grid.flatMap((row, rowIndex) =>
    row.map((token, colIndex) => {
      const id = `${rowIndex}-${colIndex}`
      const baseTone = toneMap[token] ?? 'policy-grid__cell--move'
      const isVisited = visitedKeys.value.has(id)
      const isCurrent = currentKey.value === id
      const traceStep = pathStepMap.value.get(id)

      return {
        id,
        label: traceStep && isVisited ? `${traceStep}` : labelMap[token] ?? token,
        classes: [
          'policy-grid__cell',
          baseTone,
          isVisited ? 'policy-grid__cell--visited' : '',
          isCurrent ? 'policy-grid__cell--current' : '',
        ],
      }
    }),
  ),
)
</script>

<template>
  <div class="policy-grid" :style="{ '--grid-size': `${Math.max(grid.length, 1)}` }">
    <div v-for="cell in cells" :key="cell.id" :class="cell.classes">
      {{ cell.label }}
    </div>
  </div>
</template>
