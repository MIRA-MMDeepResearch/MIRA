from .base import BaseReactAgent
from .browser import BrowserReactAgent

REGISTRY = {
    "search_open_find": BrowserReactAgent,
}

__all__ = ["BaseReactAgent", "BrowserReactAgent", "REGISTRY", "load_agent"]


def load_agent(cfg):
    variant = getattr(cfg, "model_variant", "") or ""
    agent_cls = REGISTRY.get(variant) if variant else BaseReactAgent
    if agent_cls is None:
        raise SystemExit(f"[load_agent] unknown variant {variant!r}; known: {sorted(REGISTRY)}")
    print(f"[load_agent] {variant or '(no variant)'} -> {agent_cls.__name__}")
    return agent_cls
