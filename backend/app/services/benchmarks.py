from datetime import datetime, timezone
from uuid import uuid4

from app.repositories.experiment_store import ExperimentStore
from app.schemas.experiment import (
    BenchmarkCatalogResponse,
    BenchmarkDraft,
    BenchmarkPreset,
    BenchmarkThreshold,
    CliffWalkingConfig,
    DQNConfig,
    DQNExperimentRequest,
    FrozenLakeConfig,
    FrozenLakeRewardConfig,
    QLearningConfig,
    QLearningExperimentRequest,
    ReinforceConfig,
    ReinforceExperimentRequest,
    SARSAConfig,
    SARSAExperimentRequest,
    TrainingConfig,
    WindyGridWorldConfig,
)


def build_benchmark_catalog(store: ExperimentStore | None = None) -> BenchmarkCatalogResponse:
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
            BenchmarkPreset(
                id="teacher_cliffwalking_q_learning_baseline",
                name="教师基准｜CliffWalking 中的 Q-Learning 风险路径实验",
                description="用于 CliffWalking 任务中 Q-Learning 离策略学习表现的教师参考阈值。",
                teacher_note="可将该基准与 SARSA 结果对照，观察激进最短路径与掉落悬崖风险之间的差异。",
                request=QLearningExperimentRequest(
                    name="教师基准｜CliffWalking Q-Learning 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="cliffwalking",
                    persist_result=False,
                    env_config=CliffWalkingConfig(),
                    training=TrainingConfig(episodes=160, seed=7, trace_frequency=20),
                    algorithm_config=QLearningConfig(
                        learning_rate=0.2,
                        gamma=0.92,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.98,
                        max_steps_per_episode=100,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=-260.0,
                        help_text="平均奖励应体现出 Q-Learning 在带悬崖风险环境中的可用收敛水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.65,
                        help_text="至少应有一半以上的训练回合能够到达终点。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.9,
                        help_text="训练后期应表现出较稳定的到达终点能力。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-14.0,
                        help_text="实验中应至少出现一条接近悬崖最短路径的成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_cliffwalking_sarsa_baseline",
                name="教师基准｜CliffWalking 中的 SARSA 安全策略实验",
                description="用于 CliffWalking 任务中 SARSA 同策略学习表现的教师参考阈值。",
                teacher_note="可用来讨论 SARSA 在带探索噪声时为何更倾向于学习远离悬崖的保守策略。",
                request=SARSAExperimentRequest(
                    name="教师基准｜CliffWalking SARSA 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="cliffwalking",
                    persist_result=False,
                    env_config=CliffWalkingConfig(),
                    training=TrainingConfig(episodes=160, seed=7, trace_frequency=20),
                    algorithm_config=SARSAConfig(
                        learning_rate=0.2,
                        gamma=0.92,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.98,
                        max_steps_per_episode=100,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=-160.0,
                        help_text="平均奖励应体现出 SARSA 在悬崖环境中更稳健的在线学习表现。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.7,
                        help_text="多数训练回合应能够稳定到达终点。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.9,
                        help_text="训练后期应表现出明显高于早期的稳定成功率。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-18.0,
                        help_text="实验中应至少出现一条奖励较高的安全路径成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_cliffwalking_dqn_baseline",
                name="教师基准｜CliffWalking 中的 DQN 函数逼近实验",
                description="用于 CliffWalking 任务中 DQN 表现的教师参考阈值，适合观察经验回放与目标网络在风险环境中的效果。",
                teacher_note="该基准用于展示 DQN 在经典悬崖环境中的收敛速度和在线表现，推荐与表格型算法结果进行对照。",
                request=DQNExperimentRequest(
                    name="教师基准｜CliffWalking DQN 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="cliffwalking",
                    persist_result=False,
                    env_config=CliffWalkingConfig(),
                    training=TrainingConfig(episodes=240, seed=7, trace_frequency=20),
                    algorithm_config=DQNConfig(
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.992,
                        max_steps_per_episode=100,
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
                        min_value=-450.0,
                        help_text="平均奖励应达到当前 DQN 在默认 CliffWalking 任务中的可用学习水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.6,
                        help_text="多数训练回合应能成功到达终点，说明价值函数已经具备可用性。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.9,
                        help_text="训练后期应维持较高成功率，体现目标网络和经验回放带来的稳定性。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-14.0,
                        help_text="实验中应至少出现一条接近最短成功路径的高质量轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_windygridworld_q_learning_baseline",
                name="教师基准｜WindyGridWorld 中的 Q-Learning 风力路径实验",
                description="用于 WindyGridWorld 任务中 Q-Learning 在列风扰动下学习路径修正策略的教师参考阈值。",
                teacher_note="该基准适合观察离策略更新在风力环境中的收敛表现，以及智能体如何逐步学会抵消向上风力。",
                request=QLearningExperimentRequest(
                    name="教师基准｜WindyGridWorld Q-Learning 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="windygridworld",
                    persist_result=False,
                    env_config=WindyGridWorldConfig(),
                    training=TrainingConfig(episodes=160, seed=7, trace_frequency=20),
                    algorithm_config=QLearningConfig(
                        learning_rate=0.2,
                        gamma=0.92,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.98,
                        max_steps_per_episode=100,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=-82.0,
                        help_text="平均奖励应体现出 Q-Learning 已经能够在风力扰动环境中学到可用路径。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.5,
                        help_text="至少一半训练回合应能成功到达终点，说明策略已具备基本可用性。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.9,
                        help_text="训练后期应维持较高成功率，体现策略已经能稳定修正风力影响。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-22.0,
                        help_text="实验中应至少出现一条较高质量的抗风成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_windygridworld_sarsa_baseline",
                name="教师基准｜WindyGridWorld 中的 SARSA 风力路径实验",
                description="用于 WindyGridWorld 任务中 SARSA 在风力扰动下学习在线路径修正策略的教师参考阈值。",
                teacher_note="该基准适合与 Q-Learning 结果对照，讨论同策略更新在风力环境中的在线学习特点。",
                request=SARSAExperimentRequest(
                    name="教师基准｜WindyGridWorld SARSA 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="windygridworld",
                    persist_result=False,
                    env_config=WindyGridWorldConfig(),
                    training=TrainingConfig(episodes=160, seed=7, trace_frequency=20),
                    algorithm_config=SARSAConfig(
                        learning_rate=0.2,
                        gamma=0.92,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.98,
                        max_steps_per_episode=100,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=-82.0,
                        help_text="平均奖励应体现出 SARSA 已经能够稳定适应默认风力模式。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.5,
                        help_text="多数训练回合应能成功抵达终点，说明在线更新已形成有效路径。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.9,
                        help_text="训练后期应表现出较高且稳定的到达终点能力。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-22.0,
                        help_text="实验中应至少出现一条较短的抗风成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_windygridworld_dqn_baseline",
                name="教师基准｜WindyGridWorld 中的 DQN 函数逼近实验",
                description="用于 WindyGridWorld 任务中 DQN 在列风扰动环境下学习动作价值函数的教师参考阈值。",
                teacher_note="该基准适合观察函数逼近方法在风力环境中的收敛速度，并与表格型方法进行对照。",
                request=DQNExperimentRequest(
                    name="教师基准｜WindyGridWorld DQN 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="windygridworld",
                    persist_result=False,
                    env_config=WindyGridWorldConfig(),
                    training=TrainingConfig(episodes=240, seed=7, trace_frequency=20),
                    algorithm_config=DQNConfig(
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.992,
                        max_steps_per_episode=100,
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
                        min_value=-55.0,
                        help_text="平均奖励应达到当前 DQN 在默认 WindyGridWorld 任务中的可用学习水平。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.68,
                        help_text="多数训练回合应能成功抵达终点，说明网络已经学到稳定的抗风策略。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.95,
                        help_text="训练后期应接近稳定成功，体现经验回放和目标网络带来的收益。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=-18.0,
                        help_text="实验中应至少出现一条较优的风力修正成功轨迹。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_frozenlake_q_learning_baseline",
                name="教师基准｜FrozenLake 中的 Q-Learning 稀疏奖励探索实验",
                description="用于冰湖环境中 Q-Learning 在稀疏奖励和轻度滑移扰动下学习路径策略的教师参考阈值。",
                teacher_note="该基准适当降低了滑移概率，并加入小幅冰洞惩罚，便于学生观察表格型方法在冰湖环境中的探索过程。",
                request=QLearningExperimentRequest(
                    name="教师基准｜FrozenLake Q-Learning 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="frozenlake",
                    persist_result=False,
                    env_config=FrozenLakeConfig(
                        slip_probability=0.1,
                        rewards=FrozenLakeRewardConfig(
                            step_penalty=0.0,
                            goal_reward=1.0,
                            wall_penalty=0.0,
                            hole_penalty=-0.2,
                        ),
                    ),
                    training=TrainingConfig(episodes=800, seed=7, trace_frequency=20),
                    algorithm_config=QLearningConfig(
                        learning_rate=0.2,
                        gamma=0.99,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.996,
                        max_steps_per_episode=80,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=0.24,
                        help_text="在稀疏奖励和随机滑移条件下，平均奖励应保持为正，说明已学到可用路径。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.34,
                        help_text="训练过程中应在相当比例的回合内成功到达终点，而不是仅靠偶然成功一次。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.55,
                        help_text="训练后期的表现应显示策略已经围绕一条可行路线逐步稳定下来。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=1.0,
                        help_text="采样轨迹中至少应出现一条成功到达终点的完整路径。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_frozenlake_sarsa_baseline",
                name="教师基准｜FrozenLake 中的 SARSA 滑移策略对比实验",
                description="用于冰湖环境中 SARSA 在稀疏奖励和随机滑移下学习稳定路径的教师参考阈值。",
                teacher_note="该基准适合和 Q-Learning 对照，观察同策略更新在滑移环境中是否会形成更保守的路径偏好。",
                request=SARSAExperimentRequest(
                    name="教师基准｜FrozenLake SARSA 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="frozenlake",
                    persist_result=False,
                    env_config=FrozenLakeConfig(
                        slip_probability=0.1,
                        rewards=FrozenLakeRewardConfig(
                            step_penalty=0.0,
                            goal_reward=1.0,
                            wall_penalty=0.0,
                            hole_penalty=-0.2,
                        ),
                    ),
                    training=TrainingConfig(episodes=800, seed=7, trace_frequency=20),
                    algorithm_config=SARSAConfig(
                        learning_rate=0.2,
                        gamma=0.99,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.996,
                        max_steps_per_episode=80,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=0.36,
                        help_text="平均奖励应明显高于纯探索状态，说明策略已能在稀疏奖励环境中稳定积累收益。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.46,
                        help_text="应多次稳定到达终点，而不是只出现零散的幸运成功回合。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.55,
                        help_text="训练后期的成功率应持续高于早期探索阶段，体现策略逐步收敛。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=1.0,
                        help_text="采样轨迹中至少应出现一条成功穿越冰湖的完整路径。",
                    ),
                ],
            ),
            BenchmarkPreset(
                id="teacher_frozenlake_dqn_baseline",
                name="教师基准｜FrozenLake 中的 DQN 稀疏价值学习实验",
                description="用于冰湖环境中 DQN 在稀疏奖励和随机滑移条件下学习动作价值函数的教师参考阈值。",
                teacher_note="该基准适合讨论函数逼近方法在小型离散环境中的学习表现，以及滑移噪声对样本效率的影响。",
                request=DQNExperimentRequest(
                    name="教师基准｜FrozenLake DQN 实验",
                    submitted_by="课程教师",
                    submission_role="teacher",
                    environment_id="frozenlake",
                    persist_result=False,
                    env_config=FrozenLakeConfig(
                        slip_probability=0.1,
                        rewards=FrozenLakeRewardConfig(
                            step_penalty=0.0,
                            goal_reward=1.0,
                            wall_penalty=0.0,
                            hole_penalty=-0.2,
                        ),
                    ),
                    training=TrainingConfig(episodes=260, seed=7, trace_frequency=20),
                    algorithm_config=DQNConfig(
                        learning_rate=0.001,
                        gamma=0.95,
                        epsilon_start=1.0,
                        epsilon_min=0.05,
                        epsilon_decay=0.992,
                        max_steps_per_episode=80,
                        batch_size=16,
                        replay_buffer_size=500,
                        target_sync_interval=20,
                        warmup_steps=32,
                        hidden_dim=32,
                    ),
                ),
                thresholds=[
                    BenchmarkThreshold(
                        metric_id="average_reward",
                        label="平均奖励",
                        min_value=0.09,
                        help_text="即使存在稀疏奖励和滑移噪声，平均奖励也应明显为正，说明价值函数已具备可用性。",
                    ),
                    BenchmarkThreshold(
                        metric_id="success_rate",
                        label="成功率",
                        min_value=0.22,
                        help_text="可用的 DQN 策略应在可观比例的训练回合中成功完成任务。",
                    ),
                    BenchmarkThreshold(
                        metric_id="stable_success_rate",
                        label="稳定窗口成功率",
                        min_value=0.65,
                        help_text="训练后期的表现应明显优于前期学习阶段，并维持相对稳定。",
                    ),
                    BenchmarkThreshold(
                        metric_id="best_reward",
                        label="最佳奖励",
                        min_value=1.0,
                        help_text="采样轨迹中至少应出现一条成功到达终点的高质量路径。",
                    ),
                ],
            ),
        ]
    )


def get_benchmark_preset(benchmark_id: str, store: ExperimentStore | None = None) -> BenchmarkPreset:
    for benchmark in _list_benchmarks(store):
        if benchmark.id == benchmark_id:
            return benchmark
    raise ValueError(f"未知的教师基准：{benchmark_id}")


def create_benchmark(store: ExperimentStore, draft: BenchmarkDraft) -> BenchmarkPreset:
    timestamp = datetime.now(timezone.utc).isoformat()
    benchmark = BenchmarkPreset(
        id=f"custom_benchmark_{uuid4().hex[:10]}",
        name=draft.name,
        description=draft.description,
        teacher_note=draft.teacher_note,
        request=_normalize_benchmark_request(draft),
        thresholds=draft.thresholds,
        is_builtin=False,
        created_at=timestamp,
        updated_at=timestamp,
    )
    store.save_benchmark(benchmark)
    return benchmark


def update_benchmark(store: ExperimentStore, benchmark_id: str, draft: BenchmarkDraft) -> BenchmarkPreset:
    existing = get_benchmark_preset(benchmark_id, store)
    if existing.is_builtin:
        raise ValueError("内置教师基准不支持修改")

    benchmark = BenchmarkPreset(
        id=existing.id,
        name=draft.name,
        description=draft.description,
        teacher_note=draft.teacher_note,
        request=_normalize_benchmark_request(draft),
        thresholds=draft.thresholds,
        is_builtin=False,
        created_at=existing.created_at,
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    store.save_benchmark(benchmark)
    return benchmark


def delete_benchmark(store: ExperimentStore, benchmark_id: str) -> None:
    existing = get_benchmark_preset(benchmark_id, store)
    if existing.is_builtin:
        raise ValueError("内置教师基准不支持删除")
    store.delete_benchmark(benchmark_id)


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
    if result.request.environment_id != benchmark.request.environment_id:
        return False
    if not matches_environment_config(result.request.env_config, benchmark.request.env_config):
        return False
    if result.request.training.episodes < benchmark.request.training.episodes:
        return False
    return all(get_summary_metric(result, threshold.metric_id) >= threshold.min_value for threshold in benchmark.thresholds)


def matches_environment_config(left_env, right_env) -> bool:
    return _normalize_environment_config(left_env) == _normalize_environment_config(right_env)


def _normalize_environment_config(env_config) -> dict:
    payload = env_config.model_dump(mode="json")
    for key in ("obstacles", "traps", "cliffs", "holes"):
        if key in payload:
            payload[key] = sorted(payload[key], key=lambda cell: (cell["row"], cell["col"]))
    return payload


def _list_benchmarks(store: ExperimentStore | None = None) -> list[BenchmarkPreset]:
    benchmarks = list(build_benchmark_catalog().benchmarks)
    if store is not None:
        benchmarks.extend(store.list_benchmarks())
    return benchmarks


def _normalize_benchmark_request(draft: BenchmarkDraft):
    payload = draft.request.model_dump(mode="json")
    payload["name"] = draft.name
    payload["submitted_by"] = "课程教师"
    payload["submission_role"] = "teacher"
    payload["persist_result"] = False
    payload["assignment_id"] = None
    payload["assignment_title"] = None
    return type(draft.request).model_validate(payload)
