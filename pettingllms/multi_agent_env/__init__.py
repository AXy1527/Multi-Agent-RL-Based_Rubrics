# Existing imports...
from pettingllms. multi_agent_env.base. env import Env
from pettingllms.multi_agent_env. base.agent import Agent

# Import existing environments
from pettingllms.multi_agent_env.math import MathEnv, MathEnvBatch
from pettingllms.multi_agent_env.search import SearchEnv, SearchEnvBatch
# pettingllms/multi_agent_env/__init__.py
from pettingllms.multi_agent_env.deep_search import DeepSearchEnv, DeepSearchEnvBatch

# Import Deep Search environment
from pettingllms.multi_agent_env.deep_search import (
    DeepSearchEnv,
    DeepSearchEnvBatch,
    QAgent,
    RAgent,
    AAgent,
    RMAgent,
)

# Environment registry
ENV_REGISTRY = {
    "math_env": {
        "env_class": MathEnvBatch,
    },
    "search_env": {
        "env_class": SearchEnvBatch,
    },
    "deep_search_env": {
        "env_class": DeepSearchEnvBatch,
        "agent_classes": {
            "q_agent": QAgent,
            "r_agent": RAgent,
            "a_agent": AAgent,
        },
        "rm_agent_class": RMAgent,
        "execution_engine": "pettingllms. multi_agent_env. deep_search.deep_search_execution_engine.DeepSearchExecutionEngine",
    },
}

def get_env_class(env_name: str):
    """Get environment class by name."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}.  Available: {list(ENV_REGISTRY. keys())}")
    return ENV_REGISTRY[env_name]["env_class"]

def get_agent_classes(env_name: str):
    """Get agent classes for an environment."""
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}")
    return ENV_REGISTRY[env_name]. get("agent_classes", {})

__all__ = [
    "Env",
    "Agent",
    "DeepSearchEnv",
    "DeepSearchEnvBatch",
    "QAgent",
    "RAgent",
    "AAgent",
    "RMAgent",
    "ENV_REGISTRY",
    "get_env_class",
    "get_agent_classes",
]