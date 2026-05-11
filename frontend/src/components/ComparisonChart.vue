<script setup lang="ts">
import { computed } from 'vue'

import type { ComparisonSeries } from '../types'

const props = defineProps<{
  title: string
  series: ComparisonSeries[]
}>()

const viewWidth = 680
const viewHeight = 260
const padding = 26

const normalizedSeries = computed(() => {
  if (props.series.length === 0) {
    return []
  }

  const allPoints = props.series.flatMap((item) => item.points)
  const xValues = allPoints.map((item) => item.x)
  const yValues = allPoints.map((item) => item.y)

  const minX = Math.min(...xValues)
  const maxX = Math.max(...xValues)
  const minY = Math.min(...yValues)
  const maxY = Math.max(...yValues)

  const safeXRange = Math.max(1, maxX - minX)
  const safeYRange = Math.max(1e-6, maxY - minY)

  return props.series.map((item) => ({
    ...item,
    polyline: item.points
      .map((point) => {
        const x = padding + ((point.x - minX) / safeXRange) * (viewWidth - padding * 2)
        const y = viewHeight - padding - ((point.y - minY) / safeYRange) * (viewHeight - padding * 2)
        return `${x},${y}`
      })
      .join(' '),
  }))
})
</script>

<template>
  <section class="metric-chart comparison-chart">
    <header class="metric-chart__header">
      <div>
        <p class="metric-chart__eyebrow">对比分析</p>
        <h3>{{ title }}</h3>
      </div>
      <div class="comparison-chart__legend">
        <span v-for="item in normalizedSeries" :key="item.runId" class="comparison-chart__legend-item">
          <i :style="{ background: item.color }"></i>
          {{ item.label }}
        </span>
      </div>
    </header>

    <svg
      class="metric-chart__surface comparison-chart__surface"
      :viewBox="`0 0 ${viewWidth} ${viewHeight}`"
      preserveAspectRatio="none"
      role="img"
      :aria-label="`${title} 对比图`"
    >
      <line :x1="padding" :x2="viewWidth - padding" :y1="padding" :y2="padding" class="metric-chart__grid" />
      <line
        :x1="padding"
        :x2="viewWidth - padding"
        :y1="viewHeight / 2"
        :y2="viewHeight / 2"
        class="metric-chart__grid metric-chart__grid--middle"
      />
      <line
        :x1="padding"
        :x2="viewWidth - padding"
        :y1="viewHeight - padding"
        :y2="viewHeight - padding"
        class="metric-chart__grid"
      />
      <polyline
        v-for="item in normalizedSeries"
        :key="item.runId"
        :points="item.polyline"
        fill="none"
        :stroke="item.color"
        stroke-width="3"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
    </svg>
  </section>
</template>
