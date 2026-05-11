from __future__ import annotations

from app.rl.envs.base import DiscreteGridEnvironment
from app.rl.envs.cliffwalking import CliffWalkingEnv
from app.rl.envs.gridworld import GridWorldEnv
from app.rl.envs.windygridworld import WindyGridWorldEnv
from app.schemas.experiment import ExperimentRequestType


def create_environment(request: ExperimentRequestType) -> DiscreteGridEnvironment:
    if request.environment_id == "gridworld":
        return GridWorldEnv(request.env_config)
    if request.environment_id == "cliffwalking":
        return CliffWalkingEnv(request.env_config)
    if request.environment_id == "windygridworld":
        return WindyGridWorldEnv(request.env_config)
    raise ValueError(f"Unsupported environment_id: {request.environment_id}")
