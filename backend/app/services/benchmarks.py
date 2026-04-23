from app.schemas.experiment import (
    BenchmarkCatalogResponse,
    BenchmarkPreset,
    BenchmarkThreshold,
    DQNConfig,
    DQNExperimentRequest,
    QLearningConfig,
    QLearningExperimentRequest,
    TrainingConfig,
)


def build_benchmark_catalog() -> BenchmarkCatalogResponse:
    return BenchmarkCatalogResponse(
        benchmarks=[
            BenchmarkPreset(
                id="teacher_gridworld_q_learning_baseline",
                name="Teacher Baseline | Q-Learning Fundamentals",
                description="Baseline discrete-control task for introductory reinforcement learning labs in GridWorld.",
                teacher_note=(
                    "Students should first reach this baseline before changing the reward design or exploration schedule."
                ),
                request=QLearningExperimentRequest(
                    name="Teacher Baseline | Q-Learning Fundamentals",
                    submitted_by="Course Instructor",
                    submission_role="teacher",
                    persist_result=False,
                    training=TrainingConfig(episodes=120, seed=7, trace_frequency=20),
                    algorithm_config=QLearningConfig(
                        learning_rate=0.2,
                        gamma=0.92,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.98,
                        max_steps_per_episode=80,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="Average Reward",
                        min_value=-16.0,
                        help_text="The overall reward trend should recover from early exploration noise.",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="Success Rate",
                        min_value=0.6,
                        help_text="At least 60% of episodes should reach the goal.",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="Stable Window Success",
                        min_value=0.7,
                        help_text="The recent evaluation window should show sustained convergence.",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="Best Reward",
                        min_value=8.0,
                        help_text="The learned policy should produce at least one strong successful trajectory.",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_gridworld_dqn_baseline",
                name="Teacher Baseline | DQN Approximation",
                description="Neural value-function baseline for comparing tabular and deep RL behavior in the same maze.",
                teacher_note=(
                    "Use this preset when students need to compare DQN stability with the tabular baseline under a fixed seed."
                ),
                request=DQNExperimentRequest(
                    name="Teacher Baseline | DQN Approximation",
                    submitted_by="Course Instructor",
                    submission_role="teacher",
                    persist_result=False,
                    training=TrainingConfig(episodes=120, seed=7, trace_frequency=20),
                    algorithm_config=DQNConfig(
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.992,
                        max_steps_per_episode=80,
                        batch_size=32,
                        replay_buffer_size=2000,
                        target_sync_interval=40,
                        warmup_steps=80,
                        hidden_dim=64,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="Average Reward",
                        min_value=-32.0,
                        help_text="The replay-based agent should avoid severe collapse across the full run.",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="Success Rate",
                        min_value=0.35,
                        help_text="The agent should solve the maze in at least one third of all episodes.",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="Stable Window Success",
                        min_value=0.7,
                        help_text="Late-stage behavior should be meaningfully better than the early random phase.",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="Best Reward",
                        min_value=8.0,
                        help_text="The model should discover at least one efficient goal-reaching trajectory.",
                    ),
                ],
            ),
        ]
    )


def get_benchmark_preset(benchmark_id: str) -> BenchmarkPreset:
    for benchmark in build_benchmark_catalog().benchmarks:
        if benchmark.id == benchmark_id:
            return benchmark
    raise ValueError(f"Benchmark preset not found: {benchmark_id}")


def get_summary_metric(result, metric_id: str) -> float:
    if metric_id == "average_reward":
        return result.summary.average_reward
    if metric_id == "best_reward":
        return result.summary.best_reward
    if metric_id == "success_rate":
        return result.summary.success_rate
    return result.summary.stable_success_rate


def passes_benchmark_preset(result, benchmark: BenchmarkPreset) -> bool:
    if result.request.algorithm_id != benchmark.request.algorithm_id:
        return False
    if result.request.env_config.size != benchmark.request.env_config.size:
        return False
    if result.request.training.episodes < benchmark.request.training.episodes:
        return False
    return all(get_summary_metric(result, threshold.metric_id) >= threshold.min_value for threshold in benchmark.thresholds)
