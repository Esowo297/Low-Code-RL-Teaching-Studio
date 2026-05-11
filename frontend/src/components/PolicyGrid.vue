<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  grid: string[][]
}>()

const labelMap: Record<string, string> = {
  U: '上',
  R: '右',
  D: '下',
  L: '左',
  START: '起',
  GOAL: '终',
  BLOCK: '障',
  TRAP: '陷',
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

const cells = computed(() =>
  props.grid.flatMap((row, rowIndex) =>
    row.map((token, colIndex) => ({
      id: `${rowIndex}-${colIndex}`,
      token,
      label: labelMap[token] ?? token,
      tone: toneMap[token] ?? 'policy-grid__cell--move',
    })),
  ),
)
</script>

<template>
  <div class="policy-grid" :style="{ '--grid-size': `${Math.max(grid.length, 1)}` }">
    <div v-for="cell in cells" :key="cell.id" class="policy-grid__cell" :class="cell.tone">
      {{ cell.label }}
    </div>
  </div>
</template>
