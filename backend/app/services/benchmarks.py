from app.schemas.experiment import (
    BenchmarkCatalogResponse,
    BenchmarkPreset,
    BenchmarkThreshold,
    DQNConfig,
    DQNExperimentRequest,
    QLearningConfig,
    QLearningExperimentRequest,
    ReinforceConfig,
    ReinforceExperimentRequest,
    SARSAConfig,
    SARSAExperimentRequest,
    TrainingConfig,
)


def build_benchmark_catalog() -> BenchmarkCatalogResponse:
    return BenchmarkCatalogResponse(
        benchmarks=[
            BenchmarkPreset(
                id="teacher_gridworld_q_learning_baseline",
                name="教师基准｜Q-Learning基础实验",
                description="用于默认 GridWorld 任务中 Q-Learning 入门实验的参考阈值。",
                teacher_note="可使用该基准判断学生是否达到了较稳定的入门实验结果。",
                request=QLearningExperimentRequest(
                    name="教师基准｜Q-Learning实验",
                    submitted_by="课程教师",
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
                        label="平均奖励",
                        min_value=-16.0,
                        help_text="整体平均奖励应达到预期的入门实验水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.6,
                        help_text="至少 60% 的训练轮次应成功到达目标状态。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.7,
                        help_text="训练后期的表现应体现出较稳定的策略。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=8.0,
                        help_text="实验中应至少产生一条高质量的成功路径。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_gridworld_sarsa_baseline",
                name="教师基准｜SARSA对比实验",
                description="用于默认 GridWorld 任务中 SARSA 同策略实验的参考阈值。",
                teacher_note="在相似条件下比较 SARSA 与 Q-Learning 时可采用该基准。",
                request=SARSAExperimentRequest(
                    name="教师基准｜SARSA实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    persist_result=False,
                    training=TrainingConfig(episodes=120, seed=7, trace_frequency=20),
                    algorithm_config=SARSAConfig(
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
                        label="平均奖励",
                        min_value=-18.0,
                        help_text="平均奖励应体现出同策略方法的可用解水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.5,
                        help_text="至少一半训练轮次应成功到达目标状态。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.65,
                        help_text="训练后期表现应显示出相对稳定的策略。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=8.0,
                        help_text="实验中应至少出现一条较优的成功路径。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_gridworld_dqn_baseline",
                name="教师基准｜DQN逼近实验",
                description="用于包含经验回放和目标网络同步的 DQN 实验参考阈值。",
                teacher_note="可使用该基准评估函数逼近能力和训练稳定性表现。",
                request=DQNExperimentRequest(
                    name="教师基准｜DQN实验",
                    submitted_by="课程教师",
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
                        label="平均奖励",
                        min_value=-32.0,
                        help_text="平均奖励应达到该深度值函数实验的预期水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.35,
                        help_text="实验应达到具有参考意义的成功轮次比例。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.7,
                        help_text="训练后期成功率应体现出策略逐步改善的趋势。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=8.0,
                        help_text="实验中应至少观察到一次高奖励成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_gridworld_reinforce_baseline",
                name="教师基准｜REINFORCE策略梯度实验",
                description="用于默认 GridWorld 任务中 REINFORCE 入门实验的参考阈值。",
                teacher_note="可使用该基准讨论策略梯度行为及完整回合更新特征。",
                request=ReinforceExperimentRequest(
                    name="教师基准｜REINFORCE实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    persist_result=False,
                    training=TrainingConfig(episodes=120, seed=7, trace_frequency=20),
                    algorithm_config=ReinforceConfig(
                        learning_rate=0.01,
                        gamma=0.95,
                        max_steps_per_episode=80,
                        hidden_dim=64,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=-30.0,
                        help_text="平均奖励应体现出可用的策略梯度学习趋势。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.3,
                        help_text="实验应达到具有参考意义的成功轮次数量。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.45,
                        help_text="训练后期表现应体现出策略的初步稳定化。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=5.0,
                        help_text="实验中应至少产生一条具有潜力的成功路径。",
                    ),
                ],
            ),
        ]
    )


def get_benchmark_preset(benchmark_id: str) -> BenchmarkPreset:
    for benchmark in build_benchmark_catalog().benchmarks:
        if benchmark.id == benchmark_id:
            return benchmark
    raise ValueError(f"未知的教师基准：{benchmark_id}")


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
