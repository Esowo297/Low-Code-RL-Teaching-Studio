from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from app.schemas.experiment import BenchmarkPreset, BenchmarkThreshold, ExperimentReportRequest, ExperimentResult
from app.services.benchmarks import get_benchmark_preset, get_summary_metric, passes_benchmark_preset


def render_experiment_report(request: ExperimentReportRequest) -> str:
    result = request.result
    benchmark = get_benchmark_preset(request.benchmark_id) if request.benchmark_id else None
    generated_at = datetime.now(timezone.utc).isoformat()

    sections = [
        _render_title(result),
        "## Report Metadata",
        f"- Generated At (UTC): {generated_at}",
        f"- Experiment Run ID: {result.run_id}",
        f"- Experiment Created At (UTC): {result.created_at}",
        f"- Execution Status: {result.status}",
        f"- Submitted By: {result.request.submitted_by} ({result.request.submission_role})",
        f"- Assignment: {result.request.assignment_title or 'Independent Experiment'}",
        "",
        "## Experiment Configuration",
        *list(_render_configuration(result)),
        "",
        "## Training Summary",
        *list(_render_summary(result)),
        "",
        "## Sampled Path Trace",
        *_render_trace(result),
    ]

    if benchmark is not None:
        sections.extend(
            [
                "",
                "## Teacher Benchmark Evaluation",
                f"- Benchmark: {benchmark.name}",
                f"- Description: {benchmark.description}",
                f"- Teacher Note: {benchmark.teacher_note}",
                *_render_benchmark_evaluation(result, benchmark),
            ]
        )

    sections.extend(
        [
            "",
            "## Interpretation Notes",
            *_render_interpretation(result, benchmark),
        ]
    )

    return "\n".join(sections).strip() + "\n"


def _render_title(result: ExperimentResult) -> str:
    return f"# Experiment Report | {result.request.name}"


def _render_configuration(result: ExperimentResult) -> Iterable[str]:
    request = result.request
    env_config = request.env_config
    training = request.training
    algorithm_config = request.algorithm_config

    yield f"- Environment: {request.environment_id}"
    yield f"- Algorithm: {request.algorithm_id}"
    yield f"- Grid Size: {env_config.size} x {env_config.size}"
    yield f"- Start Cell: ({env_config.start.row}, {env_config.start.col})"
    yield f"- Goal Cell: ({env_config.goal.row}, {env_config.goal.col})"
    yield f"- Obstacles: {_format_cell_list(env_config.obstacles)}"
    yield f"- Traps: {_format_cell_list(env_config.traps)}"
    yield (
        "- Reward Design: "
        f"step {env_config.rewards.step_penalty}, "
        f"goal {env_config.rewards.goal_reward}, "
        f"wall {env_config.rewards.wall_penalty}, "
        f"trap {env_config.rewards.trap_penalty}"
    )
    yield f"- Episodes: {training.episodes}"
    yield f"- Trace Frequency: {training.trace_frequency}"
    yield f"- Random Seed: {training.seed}"
    yield "- Algorithm Parameters:"
    for key, value in algorithm_config.model_dump(mode="json").items():
        yield f"  - {key}: {value}"


def _render_summary(result: ExperimentResult) -> Iterable[str]:
    summary = result.summary
    total_metrics = len(result.metrics)
    last_metric = result.metrics[-1] if result.metrics else None

    yield f"- Average Reward: {summary.average_reward:.3f}"
    yield f"- Best Reward: {summary.best_reward:.3f}"
    yield f"- Success Rate: {summary.success_rate * 100:.1f}%"
    yield f"- Stable Success Rate: {summary.stable_success_rate * 100:.1f}%"
    yield f"- Recorded Episodes: {total_metrics}"
    if last_metric is not None:
        yield f"- Final Episode Reward: {last_metric.reward:.3f}"
        yield f"- Final Episode Epsilon: {last_metric.epsilon:.3f}"
        yield f"- Final Episode Success: {'Yes' if last_metric.success else 'No'}"


def _render_trace(result: ExperimentResult) -> list[str]:
    if not result.path_traces:
        return ["- No sampled path trace was recorded."]

    trace = result.path_traces[-1]
    return [
        f"- Episode: {trace.episode}",
        f"- Success: {'Yes' if trace.success else 'No'}",
        f"- Total Reward: {trace.total_reward:.3f}",
        f"- Path: {_format_cell_list(trace.path, arrow=True)}",
    ]


def _render_benchmark_evaluation(result: ExperimentResult, benchmark: BenchmarkPreset) -> list[str]:
    checks = [
        (
            "Algorithm Match",
            result.request.algorithm_id == benchmark.request.algorithm_id,
            benchmark.request.algorithm_id,
            result.request.algorithm_id,
            "Teacher baselines should be compared within the same algorithm family.",
        ),
        (
            "Grid Size Match",
            result.request.env_config.size == benchmark.request.env_config.size,
            str(benchmark.request.env_config.size),
            str(result.request.env_config.size),
            "Changing the environment size changes task difficulty and weakens direct comparison.",
        ),
        (
            "Episode Budget",
            result.request.training.episodes >= benchmark.request.training.episodes,
            f">= {benchmark.request.training.episodes}",
            str(result.request.training.episodes),
            "The student run should not use a smaller training budget than the teacher baseline.",
        ),
    ]

    rendered = []
    for label, passed, expected, actual, note in checks:
        rendered.append(f"- {'PASS' if passed else 'FAIL'} | {label}: expected {expected}, actual {actual}")
        rendered.append(f"  - Note: {note}")

    for threshold in benchmark.thresholds:
        actual_value = get_summary_metric(result, threshold.metric_id)
        passed = actual_value >= threshold.min_value
        rendered.append(
            f"- {'PASS' if passed else 'FAIL'} | {threshold.label}: "
            f"expected >= {_format_metric(threshold, threshold.min_value)}, "
            f"actual {_format_metric(threshold, actual_value)}"
        )
        rendered.append(f"  - Note: {threshold.help_text}")

    return rendered


def _render_interpretation(result: ExperimentResult, benchmark: BenchmarkPreset | None) -> list[str]:
    notes = []
    if result.summary.stable_success_rate >= 0.7:
        notes.append("- The late-stage success window indicates that the policy has entered a relatively stable regime.")
    else:
        notes.append("- The late-stage success window remains unstable, so the convergence claim should be presented conservatively.")

    if result.summary.average_reward >= 0:
        notes.append("- The average reward is non-negative, which indicates that successful trajectories compensate for exploration cost.")
    else:
        notes.append("- The average reward is still negative, suggesting that exploration cost and failed episodes remain significant.")

    if benchmark is not None:
        passed_all = _passes_benchmark(result, benchmark)
        if passed_all:
            notes.append("- The run satisfies the selected teacher benchmark and can be treated as a completed classroom baseline.")
        else:
            notes.append("- The run does not fully satisfy the selected teacher benchmark and should be described as an intermediate result.")

    return notes


def _passes_benchmark(result: ExperimentResult, benchmark: BenchmarkPreset) -> bool:
    return passes_benchmark_preset(result, benchmark)


def _format_metric(threshold: BenchmarkThreshold, value: float) -> str:
    if threshold.metric_id in {"success_rate", "stable_success_rate"}:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _format_cell_list(cells, *, arrow: bool = False) -> str:
    if not cells:
        return "None"
    separator = " -> " if arrow else ", "
    return separator.join(f"({cell.row}, {cell.col})" for cell in cells)
