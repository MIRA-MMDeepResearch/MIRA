__all__ = ["configure_engine"]


def configure_engine(cfg) -> None:
    from .tools.common import local_fetch, tool_cache
    from .tools.common.runtime import web_image_features
    from .tools.browser import session
    from .agents import base as agents_base

    tool_cache.configure(cfg)
    local_fetch.configure(cfg)
    session.configure(cfg)
    if web_image_features(cfg).reverse_image_search:
        from .tools import image_search
        image_search.configure(cfg)
    agents_base.configure_engine(cfg)
