<script setup lang="ts">
import { computed } from 'vue'

import type { EnvironmentConfig } from '../types'

const props = defineProps<{
  grid: string[][]
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
}

function windStrengthAt(colIndex: number) {
  if (props.environmentConfig?.environment_id !== 'windygridworld') {
    return 0
  }
  return props.environmentConfig.wind_strengths[colIndex] ?? 0
}

const cells = computed(() =>
  props.grid.flatMap((row, rowIndex) =>
    row.map((token, colIndex) => {
      const windStrength = windStrengthAt(colIndex)
      return {
        id: `${rowIndex}-${colIndex}`,
        token,
        label: labelMap[token] ?? token,
        tone: toneMap[token] ?? 'policy-grid__cell--move',
        windLabel: windStrength > 0 ? `W${windStrength}` : '',
        windy: windStrength > 0,
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
      <div
        v-for="cell in cells"
        :key="cell.id"
        class="policy-grid__cell"
        :class="[cell.tone, cell.windy ? 'policy-grid__cell--windy' : '']"
      >
        <span class="policy-grid__main">{{ cell.label }}</span>
        <span v-if="cell.windLabel" class="policy-grid__wind">{{ cell.windLabel }}</span>
      </div>
    </div>
  </div>
</template>
