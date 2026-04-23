# Architecture Notes

## Design Goal

The prototype focuses on the smallest system that still supports a convincing graduation-project narrative:

- low-code experiment configuration
- interactive reinforcement learning execution
- visual feedback for training outcomes
- experiment-result persistence for later comparison

## Layered Structure

### 1. Configuration Layer

The frontend exposes a form-driven workflow that replaces handwritten experiment code. Teachers or students define:

- operator identity and submission role
- teaching assignment template
- teacher benchmark preset
- environment parameters
- reward settings
- algorithm hyperparameters
- training rounds and sampling frequency

This is the current low-code entry point for the system.

### 2. Environment Layer

The backend currently provides a configurable `GridWorld` environment. The environment model is intentionally abstracted behind a dedicated module so that later work can add:

- maze navigation
- adversarial toy tasks
- custom classroom exercises

### 3. Algorithm Layer

The algorithm package currently implements Q-learning and DQN end to end. The interface boundary is designed so the next iteration can introduce policy-gradient methods without changing the API contract used by the frontend.

### 4. Training Service Layer

The training service connects the request schema, environment instance, algorithm implementation, streaming callbacks, and persistence layer. This is the main orchestration point of the backend.

The backend also exposes a lightweight benchmark service that returns teacher-defined baseline presets and threshold targets. This keeps benchmark definitions centralized while leaving pass/fail evaluation in the frontend for immediate classroom feedback.

In addition, the backend exposes built-in assignment templates that package a task description, learning goals, recommended experiment configuration, and an optional benchmark link. This provides a course-oriented layer above raw benchmark presets.

### 5. Persistence Layer

The platform now stores experiment results in SQLite and keeps the full serialized payload for later retrieval, comparison, and replay-oriented extensions. The stored record also includes submitter identity and role metadata so that teacher dashboards can aggregate student work without requiring a separate user-management subsystem. Legacy JSON result files can be imported automatically into the database.

### 6. Visualization Layer

The frontend consumes the experiment result and renders:

- live progress updates during training
- pause, resume, and cancel control feedback
- assignment templates and learning-goal summaries
- teacher benchmark presets and threshold summaries
- benchmark pass/fail evaluation for completed runs
- Markdown report export for completed runs
- teacher/student workspace metadata
- teacher analytics for algorithm distribution, assignment progress, and student activity
- step-by-step sampled trajectory replay
- reward trend
- epsilon trend
- TD-error trend
- multi-run comparison curves
- learned policy grid
- saved run history

## Why This Scope Is Appropriate For The Thesis

- It is small enough to finish within a graduation-project schedule.
- It is complete enough to support system design, implementation details, and experiment analysis.
- It leaves clear expansion points for the "future work" section.

## Recommended Thesis Evaluation

Use two categories of experiments:

### System Experiments

- response time for experiment submission
- streaming latency between episode completion and chart refresh
- training completion stability across multiple runs
- frontend rendering fluency for longer training histories

### Teaching Experiments

- average time needed to complete a standard RL experiment
- percentage of students meeting the teacher benchmark on the first attempt
- number of parameter adjustments required before reaching the benchmark
- student understanding of reward design and exploration strategy
- usability feedback through a simple questionnaire

## Role And Analytics Scope

The current prototype uses lightweight role metadata instead of a full authentication system. This is an intentional scope control choice:

- students and teachers can be distinguished in saved experiment records
- teacher-defined benchmark presets remain reusable across users
- assignment-level and benchmark-level analytics can be computed from persisted runs without introducing account management complexity

This is sufficient for a graduation-project prototype because it demonstrates teaching workflow differentiation while keeping the implementation small and testable.

## Report Export Design

The current prototype renders teaching reports as Markdown instead of PDF. This is deliberate:

- Markdown is lightweight and easy to generate from structured experiment data.
- The exported file can be inserted into thesis appendices with minimal editing.
- The same output can later be converted to HTML or PDF without redesigning the report schema.

The report currently includes:

- experiment metadata and configuration
- assignment metadata
- reward and success summary
- sampled path trace
- teacher benchmark evaluation
- short interpretation notes suitable for classroom feedback

## Suggested Next Technical Upgrade

The strongest next upgrade is no longer basic experiment orchestration, because the current prototype already supports streaming, control, persistence, comparison, assignment templates, benchmark evaluation, report export, sampled trajectory replay, and lightweight role-based analytics. The next practical step is teaching workflow completion, for example authentication, teacher-managed template authoring, richer replay annotation, or cross-assignment cohort analytics.
