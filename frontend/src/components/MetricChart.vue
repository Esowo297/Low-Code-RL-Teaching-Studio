<script setup lang="ts">
import { computed } from 'vue'

import type { ChartPoint } from '../types'

const props = withDefaults(
  defineProps<{
    title: string
    series: ChartPoint[]
    accent?: string
  }>(),
  {
    accent: '#0f5bd8',
  },
)

const viewWidth = 520
const viewHeight = 220
const padding = 22

const normalizedSeries = computed(() => {
  if (props.series.length === 0) {
    return []
  }

  const xValues = props.series.map((item) => item.x)
  const yValues = props.series.map((item) => item.y)

  const minX = Math.min(...xValues)
  const maxX = Math.max(...xValues)
  const minY = Math.min(...yValues)
  const maxY = Math.max(...yValues)

  const safeXRange = Math.max(1, maxX - minX)
  const safeYRange = Math.max(1e-6, maxY - minY)

  return props.series.map((item) => ({
    x: padding + ((item.x - minX) / safeXRange) * (viewWidth - padding * 2),
    y: viewHeight - padding - ((item.y - minY) / safeYRange) * (viewHeight - padding * 2),
  }))
})

const polylinePoints = computed(() => normalizedSeries.value.map((item) => `${item.x},${item.y}`).join(' '))
const latestValue = computed(() => props.series.at(-1)?.y.toFixed(3) ?? '--')
const minValue = computed(() => (props.series.length ? Math.min(...props.series.map((item) => item.y)).toFixed(3) : '--'))
const maxValue = computed(() => (props.series.length ? Math.max(...props.series.map((item) => item.y)).toFixed(3) : '--'))
</script>

<template>
  <section class="metric-chart">
    <header class="metric-chart__header">
      <div>
        <p class="metric-chart__eyebrow">训练指标</p>
        <h3>{{ title }}</h3>
      </div>
      <div class="metric-chart__meta">
        <span>最新值 {{ latestValue }}</span>
      </div>
    </header>

    <svg
      class="metric-chart__surface"
      :viewBox="`0 0 ${viewWidth} ${viewHeight}`"
      preserveAspectRatio="none"
      role="img"
      :aria-label="`${title} 折线图`"
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
        v-if="polylinePoints"
        :points="polylinePoints"
        fill="none"
        :stroke="accent"
        stroke-width="3"
        stroke-linecap="round"
        stroke-linejoin="round"
      />
    </svg>

    <footer class="metric-chart__footer">
      <span>最小值 {{ minValue }}</span>
      <span>最大值 {{ maxValue }}</span>
      <span>{{ series.length }} 个采样点</span>
    </footer>
  </section>
</template>
