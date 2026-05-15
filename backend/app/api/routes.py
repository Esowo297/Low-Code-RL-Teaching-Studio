import asyncio
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, WebSocket
from fastapi.responses import PlainTextResponse
from pydantic import TypeAdapter, ValidationError
from starlette.websockets import WebSocketDisconnect, WebSocketState

from app.rl.training_artifacts import ProgressUpdate, StreamController, TrainingCancelledError
from app.schemas.experiment import (
    AssignmentCatalogResponse,
    BenchmarkCatalogResponse,
    BenchmarkDraft,
    BenchmarkPreset,
    CatalogResponse,
    ClassroomAnalyticsResponse,
    ExperimentHistoryEntry,
    ExperimentReportRequest,
    ExperimentRequest,
    ExperimentResult,
)
from app.services.analytics import build_classroom_analytics
from app.services.assignments import build_assignment_catalog
from app.services.benchmarks import build_benchmark_catalog, create_benchmark, delete_benchmark, update_benchmark
from app.services.catalog import build_catalog
from app.services.reporting import render_experiment_report
from app.services.training_service import TrainingService


router = APIRouter(prefix="/api", tags=["教学平台"])
training_service = TrainingService()
experiment_request_adapter = TypeAdapter(ExperimentRequest)


@router.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/catalog", response_model=CatalogResponse)
def get_catalog() -> CatalogResponse:
    return build_catalog()


@router.get("/benchmarks", response_model=BenchmarkCatalogResponse)
def get_benchmarks() -> BenchmarkCatalogResponse:
    builtin_catalog = build_benchmark_catalog()
    return BenchmarkCatalogResponse(
        benchmarks=[*builtin_catalog.benchmarks, *training_service.store.list_benchmarks()]
    )


@router.post("/benchmarks", response_model=BenchmarkPreset)
def create_benchmark_record(request: BenchmarkDraft) -> BenchmarkPreset:
    return create_benchmark(training_service.store, request)


@router.put("/benchmarks/{benchmark_id}", response_model=BenchmarkPreset)
def update_benchmark_record(benchmark_id: str, request: BenchmarkDraft) -> BenchmarkPreset:
    try:
        return update_benchmark(training_service.store, benchmark_id, request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/benchmarks/{benchmark_id}", status_code=204)
def delete_benchmark_record(benchmark_id: str) -> None:
    try:
        delete_benchmark(training_service.store, benchmark_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/assignments", response_model=AssignmentCatalogResponse)
def get_assignments() -> AssignmentCatalogResponse:
    return build_assignment_catalog()


@router.get("/analytics/classroom", response_model=ClassroomAnalyticsResponse)
def get_classroom_analytics(
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    benchmark_id: str | None = None,
    assignment_id: str | None = None,
) -> ClassroomAnalyticsResponse:
    try:
        return build_classroom_analytics(
            training_service.store,
            limit=limit,
            benchmark_id=benchmark_id,
            assignment_id=assignment_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/experiments/history", response_model=list[ExperimentHistoryEntry])
def get_experiment_history(limit: Annotated[int, Query(ge=1, le=20)] = 8) -> list[ExperimentHistoryEntry]:
    return training_service.list_recent_runs(limit=limit)


@router.get("/experiments/{run_id}", response_model=ExperimentResult)
def get_experiment(run_id: str) -> ExperimentResult:
    try:
        return training_service.get_run(run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/experiments/run", response_model=ExperimentResult)
def run_experiment(request: ExperimentRequest) -> ExperimentResult:
    try:
        return training_service.run_experiment(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/reports/render", response_class=PlainTextResponse)
def render_report(request: ExperimentReportRequest) -> PlainTextResponse:
    try:
        report = render_experiment_report(request, training_service.store)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    filename = f"{request.result.run_id}-report.md"
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.websocket("/experiments/stream")
async def stream_experiment(websocket: WebSocket) -> None:
    await websocket.accept()

    send_lock = asyncio.Lock()
    controller = StreamController()
    run_state = {
        "run_id": "",
        "created_at": "",
        "completed_episodes": 0,
        "total_episodes": 0,
    }

    async def send_event(payload: dict) -> None:
        if websocket.client_state == WebSocketState.DISCONNECTED:
            return
        async with send_lock:
            try:
                await websocket.send_json(payload)
            except RuntimeError:
                return

    try:
        payload = await websocket.receive_json()
        request = experiment_request_adapter.validate_python(payload)

        async def emit_start(run_id: str, created_at: str, parsed_request) -> None:
            run_state["run_id"] = run_id
            run_state["created_at"] = created_at
            await send_event(
                {
                    "type": "started",
                    "run_id": run_id,
                    "created_at": created_at,
                    "request": parsed_request.model_dump(mode="json"),
                }
            )

        async def emit_progress(update: ProgressUpdate) -> None:
            run_state["completed_episodes"] = update.completed_episodes
            run_state["total_episodes"] = update.total_episodes
            await send_event(
                {
                    "type": "metric",
                    "metric": update.metric.model_dump(mode="json"),
                    "latest_trace": update.latest_trace.model_dump(mode="json") if update.latest_trace else None,
                    "completed_episodes": update.completed_episodes,
                    "total_episodes": update.total_episodes,
                    "progress": round(update.completed_episodes / update.total_episodes, 4),
                    "running_average_reward": update.running_average_reward,
                    "running_success_rate": update.running_success_rate,
                }
            )

        async def receive_controls() -> None:
            try:
                while True:
                    message = await websocket.receive_json()
                    action = message.get("type")
                    if action == "pause":
                        controller.pause()
                        await send_event({"type": "control", "state": "paused"})
                    elif action == "resume":
                        controller.resume()
                        await send_event({"type": "control", "state": "running"})
                    elif action == "cancel":
                        controller.cancel()
                        await send_event({"type": "control", "state": "cancelling"})
                        return
            except WebSocketDisconnect:
                controller.cancel()

        async def run_training() -> None:
            try:
                result = await training_service.stream_experiment(
                    request,
                    on_start=emit_start,
                    on_progress=emit_progress,
                    controller=controller,
                )
                await send_event(
                    {
                        "type": "completed",
                        "result": result.model_dump(mode="json"),
                    }
                )
            except TrainingCancelledError:
                await send_event(
                    {
                        "type": "cancelled",
                        "run_id": run_state["run_id"],
                        "completed_episodes": run_state["completed_episodes"],
                        "total_episodes": run_state["total_episodes"],
                        "message": "训练已由用户取消",
                    }
                )

        control_task = asyncio.create_task(receive_controls())
        training_task = asyncio.create_task(run_training())

        try:
            await training_task
        finally:
            controller.cancel()
            control_task.cancel()
            await asyncio.gather(control_task, return_exceptions=True)
    except ValidationError as exc:
        await send_event(
            {
                "type": "error",
                "message": exc.errors(include_url=False),
            }
        )
    except ValueError as exc:
        await send_event(
            {
                "type": "error",
                "message": str(exc),
            }
        )
    except WebSocketDisconnect:
        controller.cancel()
        return
    finally:
        if websocket.client_state != WebSocketState.DISCONNECTED:
            await websocket.close()
