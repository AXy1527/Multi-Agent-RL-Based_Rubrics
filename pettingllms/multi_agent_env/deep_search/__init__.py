from pettingllms. multi_agent_env.deep_search. deep_search_env import (
    DeepSearchEnv,
    DeepSearchEnvBatch,
    DeepSearchEnvState,
    AgentRole,
)
from pettingllms.multi_agent_env.deep_search.agents import (
    QAgent,
    RAgent,
    AAgent,
    RMAgent,
)

__all__ = [
    "DeepSearchEnv",
    "DeepSearchEnvBatch",
    "DeepSearchEnvState",
    "AgentRole",
    "QAgent",
    "RAgent",
    "AAgent",
    "RMAgent",
]

# Environment registration for the training system
ENV_REGISTRY = {
    "deep_search_env": {
        "env_class": DeepSearchEnvBatch,
        "agent_classes": {
            "q_agent": QAgent,
            "r_agent": RAgent,
            "a_agent": AAgent,
        },
        "rm_agent_class": RMAgent,
    }
}