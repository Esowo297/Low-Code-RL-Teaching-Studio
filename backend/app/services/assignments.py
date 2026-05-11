from app.schemas.experiment import (
    AssignmentCatalogResponse,
    AssignmentPreset,
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
        ]
    )


def get_assignment_preset(assignment_id: str) -> AssignmentPreset:
    for assignment in build_assignment_catalog().assignments:
        if assignment.id == assignment_id:
            return assignment
    raise ValueError(f"未知的作业模板：{assignment_id}")
