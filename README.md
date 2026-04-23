# Low-Code Reinforcement Learning Teaching Platform

This repository contains a graduation-project prototype for a low-code, interactive reinforcement learning teaching platform.

## Current Scope

- `backend/`: FastAPI service with experiment schemas, a configurable GridWorld environment, and working Q-learning and DQN training pipelines.
- `frontend/`: Vue 3 single-page interface for experiment configuration, execution, algorithm switching, and result visualization.
- `docs/`: Architecture notes that can be reused in the thesis.
- `data/`: Runtime output directory for saved experiment results.

## MVP Features

- Form-driven experiment design for a GridWorld teaching scenario
- Parameterized Q-learning and DQN configuration
- Real-time training metric streaming over WebSocket
- Pause, resume, and cancel controls for streamed training
- Training metric output including reward, epsilon, and TD error
- Teacher benchmark presets with one-click parameter loading
- Built-in teaching assignment templates with reusable task metadata
- Benchmark pass/fail evaluation against teacher-defined thresholds
- Markdown teaching-report export for completed experiments
- Teacher/student submission metadata with role-aware classroom records
- Teacher analytics panel for cohort-level run summaries, assignment progress, and benchmark progress
- Step-by-step trajectory replay for current and selected saved runs
- Multi-run comparison dashboard for saved experiments
- SQLite-backed experiment persistence with legacy JSON import
- Policy-grid rendering for post-training inspection
- Persistent experiment history for recent runs

## Recommended Thesis Path

1. Use the current implementation as the "prototype system" baseline.
2. Use the streaming execution flow as evidence for the "interactive feedback" part of the thesis.
3. Use the training-control workflow as evidence for interactive experiment orchestration.
4. Use `docs/architecture.md` as the starting point for the system-design chapter.

## Run The Backend

```powershell
cd backend
python -m pip install -e .
uvicorn app.main:app --reload
```

The API will start at `http://127.0.0.1:8000`.

Available experiment endpoints:

- `GET /api/assignments`: built-in teaching assignment templates
- `GET /api/benchmarks`: teacher-defined benchmark presets
- `GET /api/analytics/classroom`: classroom-level run aggregation and benchmark progress
- `POST /api/experiments/run`: synchronous execution
- `WS /api/experiments/stream`: real-time streamed execution
- `POST /api/reports/render`: render a benchmark-aligned Markdown experiment report

## Run The Frontend

```powershell
cd frontend
npm install
npm run dev
```

The UI will start at `http://127.0.0.1:5173`.

## Next Recommended Iterations

- Add authentication and real access control on top of the current role metadata
- Add teacher-managed assignment and template authoring beyond the built-in presets
- Add richer replay annotation and benchmark playback comparisons
- Add rubric export and multi-assignment cohort analytics
