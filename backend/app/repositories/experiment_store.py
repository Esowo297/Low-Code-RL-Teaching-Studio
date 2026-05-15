from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from pydantic import ValidationError

from app.schemas.experiment import BenchmarkPreset, ExperimentHistoryEntry, ExperimentResult


class ExperimentStore:
    def __init__(self, base_path: Path | None = None) -> None:
        project_root = Path(__file__).resolve().parents[3]
        self.storage_root = base_path or project_root / "data"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.storage_root / "experiments.db"
        self.legacy_dir = self.storage_root / "experiments"
        self._ensure_schema()
        self._import_legacy_json_runs()

    def save(self, result: ExperimentResult) -> Path:
        payload = result.model_dump(mode="json")
        request = payload["request"]
        summary = payload["summary"]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    run_id,
                    created_at,
                    name,
                    submitted_by,
                    submission_role,
                    assignment_id,
                    assignment_title,
                    environment_id,
                    algorithm_id,
                    average_reward,
                    success_rate,
                    payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["run_id"],
                    payload["created_at"],
                    request["name"],
                    request["submitted_by"],
                    request["submission_role"],
                    request.get("assignment_id"),
                    request.get("assignment_title"),
                    request["environment_id"],
                    request["algorithm_id"],
                    summary["average_reward"],
                    summary["success_rate"],
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            connection.commit()

        return self.db_path

    def list_runs(self, limit: int = 8) -> list[ExperimentHistoryEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT run_id, name, submitted_by, submission_role, assignment_id, assignment_title, created_at, environment_id, algorithm_id, average_reward, success_rate
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            ExperimentHistoryEntry(
                run_id=row["run_id"],
                name=row["name"],
                submitted_by=row["submitted_by"],
                submission_role=row["submission_role"],
                assignment_id=row["assignment_id"],
                assignment_title=row["assignment_title"],
                created_at=row["created_at"],
                environment_id=row["environment_id"],
                algorithm_id=row["algorithm_id"],
                average_reward=row["average_reward"],
                success_rate=row["success_rate"],
            )
            for row in rows
        ]

    def list_results(self, limit: int = 50) -> list[ExperimentResult]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM experiments
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [ExperimentResult.model_validate(json.loads(row["payload"])) for row in rows]

    def load(self, run_id: str) -> ExperimentResult:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM experiments WHERE run_id = ?",
                (run_id,),
            ).fetchone()

        if row is None:
            raise FileNotFoundError(f"Experiment record not found: {run_id}")

        payload = json.loads(row["payload"])
        return ExperimentResult.model_validate(payload)

    def save_benchmark(self, benchmark: BenchmarkPreset) -> Path:
        payload = benchmark.model_dump(mode="json")
        request = payload["request"]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO benchmarks (
                    benchmark_id,
                    created_at,
                    updated_at,
                    name,
                    submitted_by,
                    environment_id,
                    algorithm_id,
                    payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["id"],
                    payload.get("created_at"),
                    payload.get("updated_at"),
                    payload["name"],
                    request["submitted_by"],
                    request["environment_id"],
                    request["algorithm_id"],
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            connection.commit()

        return self.db_path

    def list_benchmarks(self) -> list[BenchmarkPreset]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload
                FROM benchmarks
                ORDER BY updated_at DESC, created_at DESC, benchmark_id DESC
                """
            ).fetchall()

        return [BenchmarkPreset.model_validate(json.loads(row["payload"])) for row in rows]

    def load_benchmark(self, benchmark_id: str) -> BenchmarkPreset:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM benchmarks WHERE benchmark_id = ?",
                (benchmark_id,),
            ).fetchone()

        if row is None:
            raise FileNotFoundError(f"Benchmark record not found: {benchmark_id}")

        return BenchmarkPreset.model_validate(json.loads(row["payload"]))

    def delete_benchmark(self, benchmark_id: str) -> None:
        with self._connect() as connection:
            rowcount = connection.execute(
                "DELETE FROM benchmarks WHERE benchmark_id = ?",
                (benchmark_id,),
            ).rowcount
            connection.commit()

        if rowcount == 0:
            raise FileNotFoundError(f"Benchmark record not found: {benchmark_id}")

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    run_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    submitted_by TEXT NOT NULL DEFAULT 'Anonymous Student',
                    submission_role TEXT NOT NULL DEFAULT 'student',
                    assignment_id TEXT,
                    assignment_title TEXT,
                    environment_id TEXT NOT NULL,
                    algorithm_id TEXT NOT NULL,
                    average_reward REAL NOT NULL,
                    success_rate REAL NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments (created_at DESC)"
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS benchmarks (
                    benchmark_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    name TEXT NOT NULL,
                    submitted_by TEXT NOT NULL DEFAULT '课程教师',
                    environment_id TEXT NOT NULL,
                    algorithm_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_benchmarks_updated_at ON benchmarks (updated_at DESC, created_at DESC)"
            )
            existing_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(experiments)").fetchall()
            }
            if "submitted_by" not in existing_columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN submitted_by TEXT NOT NULL DEFAULT 'Anonymous Student'"
                )
            if "submission_role" not in existing_columns:
                connection.execute(
                    "ALTER TABLE experiments ADD COLUMN submission_role TEXT NOT NULL DEFAULT 'student'"
                )
            if "assignment_id" not in existing_columns:
                connection.execute("ALTER TABLE experiments ADD COLUMN assignment_id TEXT")
            if "assignment_title" not in existing_columns:
                connection.execute("ALTER TABLE experiments ADD COLUMN assignment_title TEXT")
            connection.commit()

    def _import_legacy_json_runs(self) -> None:
        if not self.legacy_dir.exists():
            return

        for result_file in self.legacy_dir.glob("*.json"):
            try:
                payload = json.loads(result_file.read_text(encoding="utf-8"))
                result = ExperimentResult.model_validate(payload)
            except (json.JSONDecodeError, OSError, ValidationError):
                continue

            self.save(result)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection
