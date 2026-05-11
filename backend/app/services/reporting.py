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
        "## 报告信息",
        f"- 生成时间（UTC）：{generated_at}",
        f"- 运行编号：{result.run_id}",
        f"- 创建时间（UTC）：{result.created_at}",
        f"- 运行状态：{_format_status(result.status)}",
        f"- 提交人：{result.request.submitted_by}（{_format_role(result.request.submission_role)}）",
        f"- 所属作业：{result.request.assignment_title or '自主实验'}",
        "",
        "## 实验配置",
        *list(_render_configuration(result)),
        "",
        "## 训练摘要",
        *list(_render_summary(result)),
        "",
        "## 采样轨迹",
        *_render_trace(result),
    ]

    if benchmark is not None:
        sections.extend(
            [
                "",
                "## 教师基准评估",
                f"- 基准名称：{benchmark.name}",
                f"- 基准说明：{benchmark.description}",
                f"- 教师备注：{benchmark.teacher_note}",
                *_render_benchmark_evaluation(result, benchmark),
            ]
        )

    sections.extend(
        [
            "",
            "## 结果解读",
            *_render_interpretation(result, benchmark),
        ]
    )

    return "\n".join(sections).strip() + "\n"


def _render_title(result: ExperimentResult) -> str:
    return f"# 实验报告 | {result.request.name}"


def _render_configuration(result: ExperimentResult) -> Iterable[str]:
    request = result.request
    env_config = request.env_config
    training = request.training
    algorithm_config = request.algorithm_config

    yield f"- 实验环境：{_format_environment_id(request.environment_id)}"
    yield f"- 训练算法：{_format_algorithm_id(request.algorithm_id)}"
    yield f"- 网格规模：{env_config.size} × {env_config.size}"
    yield f"- 起点位置：({env_config.start.row}, {env_config.start.col})"
    yield f"- 目标位置：({env_config.goal.row}, {env_config.goal.col})"
    yield f"- 障碍单元：{_format_cell_list(env_config.obstacles)}"
    yield f"- 陷阱单元：{_format_cell_list(env_config.traps)}"
    yield (
        "- 奖励设置："
        f"步进 {env_config.rewards.step_penalty}，"
        f"目标 {env_config.rewards.goal_reward}，"
        f"碰壁 {env_config.rewards.wall_penalty}，"
        f"陷阱 {env_config.rewards.trap_penalty}"
    )
    yield f"- 训练轮次：{training.episodes}"
    yield f"- 轨迹采样频率：{training.trace_frequency}"
    yield f"- 随机种子：{training.seed}"
    yield "- 算法参数："
    for key, value in algorithm_config.model_dump(mode='json').items():
        yield f"  - {key}: {value}"


def _render_summary(result: ExperimentResult) -> Iterable[str]:
    summary = result.summary
    total_metrics = len(result.metrics)
    last_metric = result.metrics[-1] if result.metrics else None

    yield f"- 平均奖励：{summary.average_reward:.3f}"
    yield f"- 最佳奖励：{summary.best_reward:.3f}"
    yield f"- 成功率：{summary.success_rate * 100:.1f}%"
    yield f"- 稳定窗口成功率：{summary.stable_success_rate * 100:.1f}%"
    yield f"- 已记录轮次：{total_metrics}"
    if last_metric is not None:
        yield f"- 最后一轮奖励：{last_metric.reward:.3f}"
        yield f"- 最后一轮探索率：{last_metric.epsilon:.3f}"
        yield f"- 最后一轮是否成功：{'是' if last_metric.success else '否'}"


def _render_trace(result: ExperimentResult) -> list[str]:
    if not result.path_traces:
        return ["- 未记录采样轨迹。"]

    trace = result.path_traces[-1]
    return [
        f"- 轨迹轮次：{trace.episode}",
        f"- 是否成功：{'是' if trace.success else '否'}",
        f"- 轨迹总奖励：{trace.total_reward:.3f}",
        f"- 路径序列：{_format_cell_list(trace.path, arrow=True)}",
    ]


def _render_benchmark_evaluation(result: ExperimentResult, benchmark: BenchmarkPreset) -> list[str]:
    checks = [
        (
            "算法一致性",
            result.request.algorithm_id == benchmark.request.algorithm_id,
            _format_algorithm_id(benchmark.request.algorithm_id),
            _format_algorithm_id(result.request.algorithm_id),
            "教师基准比较应在相同算法条件下进行。",
        ),
        (
            "环境规模一致性",
            result.request.env_config.size == benchmark.request.env_config.size,
            str(benchmark.request.env_config.size),
            str(result.request.env_config.size),
            "环境规模变化会导致任务难度变化，削弱结果可比性。",
        ),
        (
            "训练轮次要求",
            result.request.training.episodes >= benchmark.request.training.episodes,
            f">= {benchmark.request.training.episodes}",
            str(result.request.training.episodes),
            "学生实验训练轮次不应低于教师基准要求。",
        ),
    ]

    rendered = []
    for label, passed, expected, actual, note in checks:
        rendered.append(f"- {'通过' if passed else '未通过'}｜{label}：期望 {expected}，实际 {actual}")
        rendered.append(f"  - 说明：{note}")

    for threshold in benchmark.thresholds:
        actual_value = get_summary_metric(result, threshold.metric_id)
        passed = actual_value >= threshold.min_value
        rendered.append(
            f"- {'通过' if passed else '未通过'}｜{threshold.label}："
            f"期望 >= {_format_metric(threshold, threshold.min_value)}，"
            f"实际 {_format_metric(threshold, actual_value)}"
        )
        rendered.append(f"  - 说明：{threshold.help_text}")

    return rendered


def _render_interpretation(result: ExperimentResult, benchmark: BenchmarkPreset | None) -> list[str]:
    notes = []
    if result.summary.stable_success_rate >= 0.7:
        notes.append("- 训练后期成功率保持在较高水平，说明当前策略已进入相对稳定阶段。")
    else:
        notes.append("- 训练后期成功率仍存在波动，说明当前策略稳定性仍需进一步提升。")

    if result.summary.average_reward >= 0:
        notes.append("- 平均奖励已达到非负水平，说明成功轨迹收益能够较好抵消探索成本。")
    else:
        notes.append("- 平均奖励仍为负值，说明探索成本和失败回合对整体结果仍有较大影响。")

    if benchmark is not None:
        if _passes_benchmark(result, benchmark):
            notes.append("- 当前实验结果已达到所选教师基准要求，可作为较稳定的课堂参考结果。")
        else:
            notes.append("- 当前实验结果尚未完全达到所选教师基准要求，更适合作为阶段性实验结果进行分析。")

    return notes


def _passes_benchmark(result: ExperimentResult, benchmark: BenchmarkPreset) -> bool:
    return passes_benchmark_preset(result, benchmark)


def _format_metric(threshold: BenchmarkThreshold, value: float) -> str:
    if threshold.metric_id in {'success_rate', 'stable_success_rate'}:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _format_cell_list(cells, *, arrow: bool = False) -> str:
    if not cells:
        return "无"
    separator = " -> " if arrow else "，"
    return separator.join(f"({cell.row}, {cell.col})" for cell in cells)


def _format_role(role: str) -> str:
    return "教师" if role == "teacher" else "学生"


def _format_status(status: str) -> str:
    if status == "completed":
        return "已完成"
    return status


def _format_algorithm_id(algorithm_id: str) -> str:
    if algorithm_id == "q_learning":
        return "Q-Learning"
    if algorithm_id == "sarsa":
        return "SARSA"
    if algorithm_id == "dqn":
        return "DQN"
    if algorithm_id == "reinforce":
        return "REINFORCE"
    return algorithm_id


def _format_environment_id(environment_id: str) -> str:
    if environment_id == "gridworld":
        return "GridWorld"
    return environment_id
