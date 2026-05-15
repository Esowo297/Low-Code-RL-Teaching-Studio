from __future__ import annotations

from collections import defaultdict

from app.repositories.experiment_store import ExperimentStore
from app.schemas.experiment import (
    AlgorithmAnalyticsEntry,
    AssignmentAnalyticsEntry,
    BenchmarkAnalyticsSummary,
    ClassroomAnalyticsResponse,
    EnvironmentAnalyticsEntry,
    ExperimentResult,
    StudentAnalyticsEntry,
)
from app.services.assignments import get_assignment_preset
from app.services.benchmarks import get_benchmark_preset, matches_environment_config, passes_benchmark_preset


def build_classroom_analytics(
    store: ExperimentStore,
    *,
    limit: int = 50,
    benchmark_id: str | None = None,
    assignment_id: str | None = None,
) -> ClassroomAnalyticsResponse:
    results = store.list_results(limit=limit)
    if assignment_id is not None:
        get_assignment_preset(assignment_id)
        results = [result for result in results if result.request.assignment_id == assignment_id]
    total_runs = len(results)
    student_results = [result for result in results if result.request.submission_role == "student"]
    teacher_results = [result for result in results if result.request.submission_role == "teacher"]
    distinct_submitters = len({result.request.submitted_by for result in results})

    algorithms = _build_algorithm_entries(results)
    assignments = _build_assignment_entries(results, store)
    environments = _build_environment_entries(results)
    benchmark_summary, student_benchmark_pass = _build_benchmark_summary(student_results, benchmark_id, store)
    students = _build_student_entries(student_results, student_benchmark_pass)

    return ClassroomAnalyticsResponse(
        total_runs=total_runs,
        student_runs=len(student_results),
        teacher_runs=len(teacher_results),
        distinct_submitters=distinct_submitters,
        average_reward=_safe_average(result.summary.average_reward for result in results),
        average_success_rate=_safe_average(result.summary.success_rate for result in results),
        benchmark=benchmark_summary,
        assignment_filter_id=assignment_id,
        algorithms=algorithms,
        assignments=assignments,
        environments=environments,
        students=students,
    )


def _build_algorithm_entries(results: list[ExperimentResult]) -> list[AlgorithmAnalyticsEntry]:
    grouped: dict[str, list[ExperimentResult]] = defaultdict(list)
    for result in results:
        grouped[result.request.algorithm_id].append(result)

    entries = [
        AlgorithmAnalyticsEntry(
            algorithm_id=algorithm_id,
            run_count=len(items),
            average_reward=_safe_average(item.summary.average_reward for item in items),
            average_success_rate=_safe_average(item.summary.success_rate for item in items),
        )
        for algorithm_id, items in grouped.items()
    ]
    return sorted(entries, key=lambda entry: (-entry.run_count, entry.algorithm_id))


def _build_benchmark_summary(
    student_results: list[ExperimentResult],
    benchmark_id: str | None,
    store: ExperimentStore,
) -> tuple[BenchmarkAnalyticsSummary | None, dict[str, int]]:
    if benchmark_id is None:
        return None, {}

    benchmark = get_benchmark_preset(benchmark_id, store)
    eligible_runs = [
        result
        for result in student_results
        if result.request.algorithm_id == benchmark.request.algorithm_id
        and result.request.environment_id == benchmark.request.environment_id
        and matches_environment_config(result.request.env_config, benchmark.request.env_config)
    ]
    pass_by_student: dict[str, int] = defaultdict(int)
    pass_count = 0
    for result in eligible_runs:
        if passes_benchmark_preset(result, benchmark):
            pass_count += 1
            pass_by_student[result.request.submitted_by] += 1

    evaluated_runs = len(eligible_runs)
    summary = BenchmarkAnalyticsSummary(
        benchmark_id=benchmark.id,
        benchmark_name=benchmark.name,
        evaluated_runs=evaluated_runs,
        pass_count=pass_count,
        pass_rate=round(pass_count / evaluated_runs, 4) if evaluated_runs else 0.0,
    )
    return summary, pass_by_student


def _build_assignment_entries(results: list[ExperimentResult], store: ExperimentStore) -> list[AssignmentAnalyticsEntry]:
    grouped: dict[tuple[str, str], list[ExperimentResult]] = defaultdict(list)
    for result in results:
        assignment_id = result.request.assignment_id
        assignment_title = result.request.assignment_title
        if not assignment_id or not assignment_title:
            continue
        grouped[(assignment_id, assignment_title)].append(result)

    entries = []
    for (assignment_id, assignment_title), items in grouped.items():
        student_items = [item for item in items if item.request.submission_role == "student"]
        benchmark_id = _resolve_assignment_benchmark_id(items)
        benchmark_pass_rate = None
        if benchmark_id and student_items:
            benchmark = get_benchmark_preset(benchmark_id, store)
            eligible_items = [
                item
                for item in student_items
                if item.request.algorithm_id == benchmark.request.algorithm_id
                and item.request.environment_id == benchmark.request.environment_id
                and matches_environment_config(item.request.env_config, benchmark.request.env_config)
            ]
            if eligible_items:
                pass_count = sum(1 for item in eligible_items if passes_benchmark_preset(item, benchmark))
                benchmark_pass_rate = round(pass_count / len(eligible_items), 4)

        entries.append(
            AssignmentAnalyticsEntry(
                assignment_id=assignment_id,
                assignment_title=assignment_title,
                run_count=len(items),
                student_runs=len(student_items),
                distinct_submitters=len({item.request.submitted_by for item in items}),
                average_reward=_safe_average(item.summary.average_reward for item in items),
                average_success_rate=_safe_average(item.summary.success_rate for item in items),
                benchmark_id=benchmark_id,
                benchmark_pass_rate=benchmark_pass_rate,
            )
        )

    return sorted(entries, key=lambda entry: (-entry.run_count, entry.assignment_title))


def _build_environment_entries(results: list[ExperimentResult]) -> list[EnvironmentAnalyticsEntry]:
    grouped: dict[str, list[ExperimentResult]] = defaultdict(list)
    for result in results:
        grouped[result.request.environment_id].append(result)

    entries = []
    for environment_id, items in grouped.items():
        student_items = [item for item in items if item.request.submission_role == "student"]
        entries.append(
            EnvironmentAnalyticsEntry(
                environment_id=environment_id,
                run_count=len(items),
                student_runs=len(student_items),
                distinct_submitters=len({item.request.submitted_by for item in items}),
                average_reward=_safe_average(item.summary.average_reward for item in items),
                average_success_rate=_safe_average(item.summary.success_rate for item in items),
                best_success_rate=max(item.summary.success_rate for item in items),
            )
        )

    return sorted(entries, key=lambda entry: (-entry.run_count, entry.environment_id))


def _build_student_entries(
    student_results: list[ExperimentResult],
    student_benchmark_pass: dict[str, int],
) -> list[StudentAnalyticsEntry]:
    grouped: dict[str, list[ExperimentResult]] = defaultdict(list)
    for result in student_results:
        grouped[result.request.submitted_by].append(result)

    entries = []
    for submitted_by, items in grouped.items():
        latest_created_at = max(item.created_at for item in items)
        benchmark_pass_count = student_benchmark_pass.get(submitted_by)
        entries.append(
            StudentAnalyticsEntry(
                submitted_by=submitted_by,
                run_count=len(items),
                average_reward=_safe_average(item.summary.average_reward for item in items),
                average_success_rate=_safe_average(item.summary.success_rate for item in items),
                best_success_rate=max(item.summary.success_rate for item in items),
                latest_created_at=latest_created_at,
                benchmark_pass_count=benchmark_pass_count if student_benchmark_pass else None,
            )
        )

    return sorted(entries, key=lambda entry: (-entry.run_count, -entry.best_success_rate, entry.submitted_by))


def _safe_average(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _resolve_assignment_benchmark_id(items: list[ExperimentResult]) -> str | None:
    assignment_id = items[0].request.assignment_id
    if not assignment_id:
        return None

    try:
        return get_assignment_preset(assignment_id).benchmark_id
    except ValueError:
        return None
