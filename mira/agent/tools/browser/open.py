import re
from typing import Union

from qwen_agent.tools.base import BaseTool, register_tool

from .session import BrowserSession, BrowserUsageError


DESCRIPTION_TEXT_ONLY = (
    "Opens a link from the current page or a fully qualified URL. Can scroll to a "
    "specific location and display a specific number of lines. Valid link ids are "
    "displayed with the formatting: 【{id}†.*】."
)

DESCRIPTION_WITH_IMAGE = (
    DESCRIPTION_TEXT_ONLY + " If a prior browser.open or browser.find observation "
    "contains a compact webpage-image marker such as [page_img_1: caption], and the "
    "question requires visual details from that webpage image, call browser.open with "
    "id=\"page_img_1\" to attach that specific image's pixels. Raw image URLs are "
    "intentionally kept out of context."
)

ID_DESCRIPTION_TEXT_ONLY = (
    "Link id from current page (integer) or fully qualified URL (string). "
    "Default is -1 (most recent page)"
)

_IMAGE_HANDLE = re.compile(r"^page_img_\d+$", re.IGNORECASE)


def image_handle(value) -> str:
    text = str(value or "").strip()
    return text if _IMAGE_HANDLE.match(text) else ""


@register_tool("browser.open", allow_overwrite=True)
class Open(BaseTool):
    name = "browser.open"
    description = DESCRIPTION_WITH_IMAGE
    parameters = {
        "type": "object",
        "properties": {
            "id": {
                "type": ["integer", "string"],
                "description": (
                    "What to open: a link id from the current page (integer), a fully "
                    "qualified URL (string), or a page_img_N handle shown in a previous "
                    "observation (string) to get that webpage image's pixels. "
                    "Default is -1 (most recent page)"
                ),
                "default": -1,
            },
            "cursor": {
                "type": "integer",
                "description": (
                    "Page cursor to operate on. If not provided, the most recent page "
                    "is implied"
                ),
                "default": -1,
            },
            "loc": {
                "type": "integer",
                "description": (
                    "Starting line number. If not provided, viewport will be positioned "
                    "at the beginning or centered on relevant passage"
                ),
                "default": -1,
            },
            "num_lines": {
                "type": "integer",
                "description": "Number of lines to display",
                "default": -1,
            },
            "view_source": {
                "type": "boolean",
                "description": "Whether to view page source",
                "default": False,
            },
            "source": {
                "type": "string",
                "description": "The source identifier (e.g., 'web')",
            },
        },
        "required": [],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        session = kwargs.get("session")
        if not isinstance(session, BrowserSession):
            return "[Tool Error] browser.open requires an active BrowserSession."
        if not isinstance(params, dict):
            return "[Tool Error] browser.open arguments must be an object."
        try:
            image_ref = str(params.get("image") or params.get("image_url") or "").strip()
            if image_ref:
                return session.open_image(
                    image_ref,
                    handle=str(params.get("image_handle") or image_ref),
                    caption=str(params.get("image_caption") or ""),
                    page_url=str(params.get("image_page_url") or ""),
                    workdir=str(kwargs.get("mm_search_workdir") or ""),
                )
            return session.open(
                id=params.get("id", -1),
                cursor=params.get("cursor", -1),
                loc=params.get("loc", -1),
                num_lines=params.get("num_lines", -1),
                view_source=bool(params.get("view_source", False)),
                web_multimodal_enabled=bool(kwargs.get("web_multimodal_enabled", False)),
                page_image_open_enabled=bool(kwargs.get("page_image_open_enabled", False)),
                image_workdir=str(kwargs.get("mm_search_workdir") or ""),
                image_start=int(kwargs.get("mm_search_start", 1)),
            )
        except (BrowserUsageError, TypeError, ValueError) as exc:
            return str(exc)


__all__ = ["Open"]
