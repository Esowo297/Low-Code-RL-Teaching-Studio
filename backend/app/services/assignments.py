from app.schemas.experiment import (
    AssignmentCatalogResponse,
    AssignmentPreset,
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


def build_assignment_catalog() -> AssignmentCatalogResponse:
    return AssignmentCatalogResponse(
        assignments=[
            AssignmentPreset(
                id="assignment_gridworld_q_learning_convergence",
                title="作业1｜GridWorld中的Q-Learning收敛实验",
                summary="观察表格型值函数方法在小规模离散环境中的收敛过程。",
                instructions=(
                    "在默认 GridWorld 环境中运行 Q-Learning，调整探索率变化策略，"
                    "比较不同参数设置对奖励增长和成功率变化的影响。"
                ),
                learning_goals=[
                    "理解 Q-Learning 中值函数更新的作用。",
                    "观察探索率变化对探索行为和收敛过程的影响。",
                    "结合奖励曲线、成功率和采样轨迹理解训练结果。",
                ],
                benchmark_id="teacher_gridworld_q_learning_baseline",
                request=QLearningExperimentRequest(
                    name="作业1｜Q-Learning实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_q_learning_convergence",
                    assignment_title="作业1｜GridWorld中的Q-Learning收敛实验",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_gridworld_sarsa_policy_comparison",
                title="作业2｜SARSA与Q-Learning对比实验",
                summary="比较同策略与离策略表格型强化学习方法的训练行为差异。",
                instructions=(
                    "在相同 GridWorld 任务中运行 SARSA，比较其奖励变化趋势、路径行为和收敛速度，"
                    "并与 Q-Learning 基准结果进行对照。"
                ),
                learning_goals=[
                    "理解同策略与离策略更新方式之间的区别。",
                    "比较相似探索设置下的路径选择差异。",
                    "分析 SARSA 在风险状态和不确定状态下的行为特点。",
                ],
                benchmark_id="teacher_gridworld_sarsa_baseline",
                request=SARSAExperimentRequest(
                    name="作业2｜SARSA实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_sarsa_policy_comparison",
                    assignment_title="作业2｜SARSA与Q-Learning对比实验",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_gridworld_dqn_comparison",
                title="作业3｜DQN稳定性与函数逼近实验",
                summary="观察深度值函数方法在相同教学环境中的训练特征。",
                instructions=(
                    "在 GridWorld 中运行 DQN，并与表格型方法对比其学习过程，"
                    "重点观察经验回放、目标网络同步和训练稳定性。"
                ),
                learning_goals=[
                    "理解强化学习中函数逼近的基本动机。",
                    "观察经验回放与目标网络更新对训练的影响。",
                    "比较深度值函数学习与表格型方法之间的差异。",
                ],
                benchmark_id="teacher_gridworld_dqn_baseline",
                request=DQNExperimentRequest(
                    name="作业3｜DQN实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_dqn_comparison",
                    assignment_title="作业3｜DQN稳定性与函数逼近实验",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_gridworld_reinforce_policy_gradient",
                title="作业4｜REINFORCE策略梯度基础实验",
                summary="在教学环境中体验直接策略优化方法的训练过程。",
                instructions=(
                    "在 GridWorld 中运行 REINFORCE，观察完整回合回报如何影响策略更新，"
                    "并将其行为与值函数方法进行比较。"
                ),
                learning_goals=[
                    "理解直接策略优化的基本思想。",
                    "观察基于完整回合回报更新的训练特征。",
                    "比较策略梯度方法与值函数方法的行为差异。",
                ],
                benchmark_id="teacher_gridworld_reinforce_baseline",
                request=ReinforceExperimentRequest(
                    name="作业4｜REINFORCE实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_reinforce_policy_gradient",
                    assignment_title="作业4｜REINFORCE策略梯度基础实验",
                    persist_result=True,
                    training=TrainingConfig(episodes=120, seed=7, trace_frequency=20),
                    algorithm_config=ReinforceConfig(
                        learning_rate=0.01,
                        gamma=0.95,
                        max_steps_per_episode=80,
                        hidden_dim=64,
                    ),
                ),
            ),
            AssignmentPreset(
                id="assignment_cliffwalking_q_learning_risk_path",
                title="作业5｜CliffWalking 中的 Q-Learning 风险路径分析",
                summary="观察 Q-Learning 在悬崖环境中如何逐步逼近最短路径，并分析探索阶段掉落悬崖带来的奖励波动。",
                instructions=(
                    "在 CliffWalking 环境中运行 Q-Learning，重点观察奖励曲线、成功率和轨迹回放，"
                    "分析其为何会倾向于学习贴近悬崖的激进路径。"
                ),
                learning_goals=[
                    "理解离策略更新在风险环境中的行为特点。",
                    "观察探索动作对悬崖任务在线表现的影响。",
                    "结合轨迹回放分析最短路径与安全性之间的权衡。",
                ],
                benchmark_id="teacher_cliffwalking_q_learning_baseline",
                request=QLearningExperimentRequest(
                    name="作业5｜CliffWalking Q-Learning 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_cliffwalking_q_learning_risk_path",
                    assignment_title="作业5｜CliffWalking 中的 Q-Learning 风险路径分析",
                    environment_id="cliffwalking",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_cliffwalking_sarsa_safe_policy",
                title="作业6｜CliffWalking 中的 SARSA 安全策略对比",
                summary="使用 SARSA 在悬崖环境中训练，比较其与 Q-Learning 在路径选择和在线稳定性上的差异。",
                instructions=(
                    "在相同 CliffWalking 任务中运行 SARSA，并与 Q-Learning 基准进行对照，"
                    "重点分析其为何更容易学到远离悬崖的保守路径。"
                ),
                learning_goals=[
                    "理解同策略更新在探索阶段的安全性优势。",
                    "比较 SARSA 与 Q-Learning 在成功率和平均奖励上的差异。",
                    "通过轨迹回放识别保守策略和激进策略的具体表现。",
                ],
                benchmark_id="teacher_cliffwalking_sarsa_baseline",
                request=SARSAExperimentRequest(
                    name="作业6｜CliffWalking SARSA 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_cliffwalking_sarsa_safe_policy",
                    assignment_title="作业6｜CliffWalking 中的 SARSA 安全策略对比",
                    environment_id="cliffwalking",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_cliffwalking_dqn_value_approximation",
                title="作业7｜CliffWalking 中的 DQN 逼近策略实验",
                summary="在经典悬崖环境中使用 DQN 学习动作价值函数，观察经验回放和目标网络对收敛稳定性的影响。",
                instructions=(
                    "在默认 CliffWalking 环境中运行 DQN，重点比较其奖励曲线、稳定窗口成功率和轨迹回放，"
                    "分析函数逼近方法在高惩罚风险环境中的优势与代价。"
                ),
                learning_goals=[
                    "理解 DQN 在离散网格环境中的价值函数逼近流程。",
                    "观察经验回放和目标网络对训练稳定性的影响。",
                    "比较 DQN 与表格型算法在悬崖环境中的在线表现差异。",
                ],
                benchmark_id="teacher_cliffwalking_dqn_baseline",
                request=DQNExperimentRequest(
                    name="作业7｜CliffWalking DQN 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_cliffwalking_dqn_value_approximation",
                    assignment_title="作业7｜CliffWalking 中的 DQN 逼近策略实验",
                    environment_id="cliffwalking",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_windygridworld_q_learning_wind_compensation",
                title="作业8｜WindyGridWorld 中的 Q-Learning 抗风路径分析",
                summary="观察 Q-Learning 如何在列风扰动环境中逐步学会补偿向上风力，并分析其路径修正过程。",
                instructions=(
                    "在默认 WindyGridWorld 环境中运行 Q-Learning，重点观察奖励曲线、成功率和策略网格，"
                    "分析智能体如何通过动作选择抵消不同列的风力影响。"
                ),
                learning_goals=[
                    "理解环境动力学变化如何影响状态转移与最优路径。",
                    "观察离策略更新在风力扰动环境中的收敛特点。",
                    "结合策略网格与轨迹回放分析抗风路径的形成过程。",
                ],
                benchmark_id="teacher_windygridworld_q_learning_baseline",
                request=QLearningExperimentRequest(
                    name="作业8｜WindyGridWorld Q-Learning 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_windygridworld_q_learning_wind_compensation",
                    assignment_title="作业8｜WindyGridWorld 中的 Q-Learning 抗风路径分析",
                    environment_id="windygridworld",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_windygridworld_sarsa_online_correction",
                title="作业9｜WindyGridWorld 中的 SARSA 在线修正策略实验",
                summary="使用 SARSA 在风力环境中进行在线学习，并比较其与 Q-Learning 在路径修正过程中的差异。",
                instructions=(
                    "在相同 WindyGridWorld 任务中运行 SARSA，并与 Q-Learning 基准进行对照，"
                    "重点分析其成功率变化、策略稳定性和抗风路径选择。"
                ),
                learning_goals=[
                    "理解同策略更新在动态转移环境中的学习特征。",
                    "比较 SARSA 与 Q-Learning 在风力环境中的路径修正差异。",
                    "通过轨迹回放识别在线学习对策略稳定性的影响。",
                ],
                benchmark_id="teacher_windygridworld_sarsa_baseline",
                request=SARSAExperimentRequest(
                    name="作业9｜WindyGridWorld SARSA 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_windygridworld_sarsa_online_correction",
                    assignment_title="作业9｜WindyGridWorld 中的 SARSA 在线修正策略实验",
                    environment_id="windygridworld",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_windygridworld_dqn_value_learning",
                title="作业10｜WindyGridWorld 中的 DQN 风力价值学习实验",
                summary="在列风扰动环境中使用 DQN 学习动作价值函数，观察函数逼近方法如何学习抗风路径。",
                instructions=(
                    "在默认 WindyGridWorld 环境中运行 DQN，比较其奖励曲线、稳定窗口成功率与策略网格，"
                    "分析经验回放和目标网络在风力环境中的作用。"
                ),
                learning_goals=[
                    "理解 DQN 在动态网格环境中的价值函数学习过程。",
                    "观察函数逼近方法对风力扰动环境的适应能力。",
                    "比较 DQN 与表格型算法在风力环境中的训练表现差异。",
                ],
                benchmark_id="teacher_windygridworld_dqn_baseline",
                request=DQNExperimentRequest(
                    name="作业10｜WindyGridWorld DQN 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_windygridworld_dqn_value_learning",
                    assignment_title="作业10｜WindyGridWorld 中的 DQN 风力价值学习实验",
                    environment_id="windygridworld",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_frozenlake_q_learning_sparse_reward",
                title="作业11｜FrozenLake 中的 Q-Learning 稀疏奖励探索",
                summary="观察表格型 Q-Learning 在冰湖环境的稀疏奖励和随机滑移条件下如何逐步找到可行路径。",
                instructions=(
                    "运行给定的冰湖环境预设，观察奖励曲线、成功率和采样轨迹，"
                    "分析滑移扰动与终止冰洞如何提高探索难度。"
                ),
                learning_goals=[
                    "理解稀疏奖励如何减慢小型离散环境中的价值传播速度。",
                    "观察随机滑移如何增加探索成本并扰动早期训练轨迹。",
                    "结合策略视图与轨迹回放分析失败回合在冰湖中的集中区域。",
                ],
                benchmark_id="teacher_frozenlake_q_learning_baseline",
                request=QLearningExperimentRequest(
                    name="作业11｜FrozenLake Q-Learning 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_frozenlake_q_learning_sparse_reward",
                    assignment_title="作业11｜FrozenLake 中的 Q-Learning 稀疏奖励探索",
                    environment_id="frozenlake",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_frozenlake_sarsa_slip_comparison",
                title="作业12｜FrozenLake 中的 SARSA 滑移策略对比",
                summary="在同一冰湖环境下比较 SARSA 与 Q-Learning，分析同策略更新是否会形成不同路径偏好。",
                instructions=(
                    "运行 SARSA 的冰湖环境预设，并与 Q-Learning 基准进行对照。"
                    "重点分析成功率稳定性、冰洞附近的路径选择，以及滑移噪声对在线更新的影响。"
                ),
                learning_goals=[
                    "比较随机转移条件下同策略与离策略学习的差异。",
                    "识别 SARSA 是否会在高风险单元附近学到更保守的绕行路线。",
                    "结合轨迹而不仅是汇总指标来解释算法差异。",
                ],
                benchmark_id="teacher_frozenlake_sarsa_baseline",
                request=SARSAExperimentRequest(
                    name="作业12｜FrozenLake SARSA 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_frozenlake_sarsa_slip_comparison",
                    assignment_title="作业12｜FrozenLake 中的 SARSA 滑移策略对比",
                    environment_id="frozenlake",
                    persist_result=True,
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
            ),
            AssignmentPreset(
                id="assignment_frozenlake_dqn_sparse_value_learning",
                title="作业13｜FrozenLake 中的 DQN 稀疏价值学习",
                summary="使用 DQN 在稀疏奖励的冰湖环境中学习动作价值，并与表格型方法进行比较。",
                instructions=(
                    "运行 DQN 的冰湖环境预设，评估函数逼近是否改善了训练后期的成功表现。"
                    "结合奖励曲线、样本效率和策略网格，分析其与表格型方法的差异。"
                ),
                learning_goals=[
                    "观察 DQN 在小型但带噪声的稀疏奖励环境中的训练表现。",
                    "比较基于经验回放的学习方式与表格型探索在同一任务中的差异。",
                    "在判断学习是否真正稳定时，同时参考基准评估和轨迹回放。",
                ],
                benchmark_id="teacher_frozenlake_dqn_baseline",
                request=DQNExperimentRequest(
                    name="作业13｜FrozenLake DQN 实验",
                    submitted_by="学生A",
                    submission_role="student",
                    assignment_id="assignment_frozenlake_dqn_sparse_value_learning",
                    assignment_title="作业13｜FrozenLake 中的 DQN 稀疏价值学习",
                    environment_id="frozenlake",
                    persist_result=True,
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
            ),
        ]
    )


def get_assignment_preset(assignment_id: str) -> AssignmentPreset:
    for assignment in build_assignment_catalog().assignments:
        if assignment.id == assignment_id:
            return assignment
    raise ValueError(f"未知的作业模板：{assignment_id}")
