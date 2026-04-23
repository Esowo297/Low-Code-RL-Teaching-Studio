from app.schemas.experiment import (
    AssignmentCatalogResponse,
    AssignmentPreset,
    DQNConfig,
    DQNExperimentRequest,
    QLearningConfig,
    QLearningExperimentRequest,
    TrainingConfig,
)


def build_assignment_catalog() -> AssignmentCatalogResponse:
    return AssignmentCatalogResponse(
        assignments=[
            AssignmentPreset(
                id="assignment_gridworld_q_learning_convergence",
                title="Assignment 1 | Q-Learning Convergence in GridWorld",
                summary="Tune a tabular agent until it reliably reaches the goal and explain how exploration decay affects convergence.",
                instructions=(
                    "Use the baseline GridWorld maze, observe the reward curve and epsilon schedule, "
                    "then explain why the learned policy converges or fails to converge."
                ),
                learning_goals=[
                    "Understand the exploration-exploitation tradeoff in tabular reinforcement learning.",
                    "Interpret reward curves and stable success rate as convergence indicators.",
                    "Compare parameter tuning decisions against a teacher baseline.",
                ],
                benchmark_id="teacher_gridworld_q_learning_baseline",
                request=QLearningExperimentRequest(
                    name="Assignment 1 | Q-Learning Convergence",
                    submitted_by="Student A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_q_learning_convergence",
                    assignment_title="Assignment 1 | Q-Learning Convergence in GridWorld",
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
                id="assignment_gridworld_dqn_comparison",
                title="Assignment 2 | DQN Stability and Comparison",
                summary="Run a neural value-function agent on the same maze and compare its stability with the tabular baseline.",
                instructions=(
                    "Execute the DQN setup on the standard GridWorld task, compare the reward and success curves "
                    "against Q-learning, and identify where approximation changes the learning dynamics."
                ),
                learning_goals=[
                    "Recognize differences between tabular and deep value-function learning.",
                    "Interpret replay memory, target network updates, and late-stage success trends.",
                    "Summarize when DQN is beneficial and when it is harder to stabilize in small tasks.",
                ],
                benchmark_id="teacher_gridworld_dqn_baseline",
                request=DQNExperimentRequest(
                    name="Assignment 2 | DQN Stability and Comparison",
                    submitted_by="Student A",
                    submission_role="student",
                    assignment_id="assignment_gridworld_dqn_comparison",
                    assignment_title="Assignment 2 | DQN Stability and Comparison",
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
        ]
    )


def get_assignment_preset(assignment_id: str) -> AssignmentPreset:
    for assignment in build_assignment_catalog().assignments:
        if assignment.id == assignment_id:
            return assignment
    raise ValueError(f"Assignment preset not found: {assignment_id}")
