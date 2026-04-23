from app.schemas.experiment import CatalogResponse, ModuleDefinition, ParameterDefinition


def build_catalog() -> CatalogResponse:
    return CatalogResponse(
        environments=[
            ModuleDefinition(
                id="gridworld",
                name="GridWorld",
                description="A classroom-friendly discrete environment for reward design, path planning, and policy inspection.",
                parameters=[
                    ParameterDefinition(
                        key="size",
                        label="Grid Size",
                        default=6,
                        min_value=4,
                        max_value=12,
                        step=1,
                        help_text="Controls the width and height of the discrete environment.",
                    ),
                    ParameterDefinition(
                        key="obstacles",
                        label="Obstacle Cells",
                        default="1:2,2:2,4:2,4:3",
                        help_text="Comma-separated row:col pairs for blocked cells.",
                    ),
                    ParameterDefinition(
                        key="traps",
                        label="Trap Cells",
                        default="2:4,4:1",
                        help_text="Comma-separated row:col pairs for terminal negative-reward cells.",
                    ),
                    ParameterDefinition(
                        key="goal_reward",
                        label="Goal Reward",
                        default=20,
                        min_value=1,
                        max_value=100,
                        step=1,
                        help_text="Reward assigned when the agent reaches the goal cell.",
                    ),
                ],
            )
        ],
        algorithms=[
            ModuleDefinition(
                id="q_learning",
                name="Q-Learning",
                description="Tabular off-policy control for discrete state and action spaces.",
                parameters=[
                    ParameterDefinition(
                        key="learning_rate",
                        label="Learning Rate",
                        default=0.2,
                        min_value=0.01,
                        max_value=1.0,
                        step=0.01,
                        help_text="Controls the size of Q-value updates.",
                    ),
                    ParameterDefinition(
                        key="gamma",
                        label="Discount Factor",
                        default=0.92,
                        min_value=0.5,
                        max_value=0.999,
                        step=0.001,
                        help_text="Weights future rewards in the Bellman target.",
                    ),
                    ParameterDefinition(
                        key="epsilon_decay",
                        label="Epsilon Decay",
                        default=0.98,
                        min_value=0.8,
                        max_value=1.0,
                        step=0.001,
                        help_text="Decays the exploration rate after each episode.",
                    ),
                ],
            ),
            ModuleDefinition(
                id="dqn",
                name="DQN",
                description="A neural value-function approximator with replay memory and a target network.",
                parameters=[
                    ParameterDefinition(
                        key="learning_rate",
                        label="Learning Rate",
                        default=0.001,
                        min_value=0.0001,
                        max_value=0.01,
                        step=0.0001,
                        help_text="Controls the optimizer step size for the neural network.",
                    ),
                    ParameterDefinition(
                        key="batch_size",
                        label="Batch Size",
                        default=32,
                        min_value=8,
                        max_value=256,
                        step=1,
                        help_text="Number of replay samples used in each optimization step.",
                    ),
                    ParameterDefinition(
                        key="replay_buffer_size",
                        label="Replay Buffer",
                        default=2000,
                        min_value=200,
                        max_value=50000,
                        step=100,
                        help_text="Capacity of the experience replay memory.",
                    ),
                    ParameterDefinition(
                        key="target_sync_interval",
                        label="Target Sync",
                        default=40,
                        min_value=1,
                        max_value=1000,
                        step=1,
                        help_text="How often to copy policy-network weights to the target network.",
                    ),
                ],
            ),
        ],
    )
