from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from .session import BrowserSession, BrowserUsageError


@register_tool("browser.find", allow_overwrite=True)
class Find(BaseTool):
    name = "browser.find"
    description = (
        "Finds exact matches of a pattern in the current page or a specified page by cursor."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The exact text pattern to search for",
            },
            "cursor": {
                "type": "integer",
                "description": (
                    "Page cursor to search in. If not provided, searches in the current page"
                ),
                "default": -1,
            },
        },
        "required": ["pattern"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        session = kwargs.get("session")
        if not isinstance(session, BrowserSession):
            return "[Tool Error] browser.find requires an active BrowserSession."
        if not isinstance(params, dict):
            return "[Tool Error] browser.find arguments must be an object."
        try:
            return session.find(
                params.get("pattern", ""),
                cursor=params.get("cursor", -1),
                web_multimodal_enabled=bool(kwargs.get("web_multimodal_enabled", False)),
                page_image_open_enabled=bool(kwargs.get("page_image_open_enabled", False)),
                image_workdir=str(kwargs.get("mm_search_workdir") or ""),
                image_start=int(kwargs.get("mm_search_start", 1)),
            )
        except (BrowserUsageError, TypeError, ValueError) as exc:
            return str(exc)


__all__ = ["Find"]
