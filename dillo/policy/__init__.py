"""ACT policy training and inference utilities."""

__all__ = ["ACTAgent", "ActionChunkingPolicy"]


def __getattr__(name):
    if name == "ACTAgent":
        from .act_agent import ACTAgent

        return ACTAgent
    if name == "ActionChunkingPolicy":
        from .act_policy import ActionChunkingPolicy

        return ActionChunkingPolicy
    raise AttributeError(name)
