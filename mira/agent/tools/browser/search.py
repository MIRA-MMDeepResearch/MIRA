from __future__ import annotations

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import List, Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from ..common.image_assets import ImageAttachment, PageImage, image_name_from_url, materialize_image
from .session import (
    BrowserResult,
    BrowserSession,
    BrowserUsageError,
    S,
    web_image_limits,
)


@register_tool("browser.search", allow_overwrite=True)
class BrowserSearch(BaseTool):
    name = "browser.search"
    description = (
        "Searches for information related to a query and displays top N results. "
        "Returns a list of search results with titles, URLs, and summaries."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "The search query string"},
            "topn": {
                "type": "integer",
                "description": "Number of results to display",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def call(self, params: Union[str, dict], **kwargs) -> str:
        session = kwargs.get("session")
        if not isinstance(session, BrowserSession):
            return "[Tool Error] browser.search requires an active BrowserSession."
        if not isinstance(params, dict):
            return "[Tool Error] browser.search arguments must be an object."
        try:
            return session.search(params.get("query"), topn=params.get("topn", 10))
        except (BrowserUsageError, TypeError, ValueError) as exc:
            return str(exc)


def fetch_result_image_attachments(
    records,
    workdir: str,
    start_index: int = 1,
    max_images: Optional[int] = None,
) -> List[ImageAttachment]:
    configured_per_call, configured_per_page, max_bytes = web_image_limits()
    max_images = min(
        configured_per_call if max_images is None else max(0, int(max_images)),
        configured_per_call,
        configured_per_page,
    )

    def download(index_record):
        index, record = index_record
        source_url = str(
            record.get("image_url")
            or record.get("imageUrl")
            or record.get("thumbnail_url")
            or ""
        )
        page_url = str(record.get("link") or record.get("url") or source_url)
        if not source_url:
            return index, None
        handle = f"_search_candidate_{index}"
        image = PageImage(
            source_url=source_url,
            page_url=page_url,
            marker="",
            original_name=image_name_from_url(source_url),
            alt=str(record.get("title") or ""),
            caption=str(record.get("source") or ""),
        )
        try:
            return index, materialize_image(
                image, workdir, handle, max_bytes=max_bytes,
                proxy_kwargs=S().proxy_kwargs,
            )
        except Exception:
            return index, None

    candidates = list(enumerate(list(records)[:configured_per_page]))
    downloaded = {}
    if candidates:
        with ThreadPoolExecutor(max_workers=min(8, len(candidates))) as executor:
            futures = [executor.submit(download, candidate) for candidate in candidates]
            for future in as_completed(futures):
                index, attachment = future.result()
                if attachment is not None:
                    downloaded[index] = attachment

    attachments: List[ImageAttachment] = []
    for index in sorted(downloaded):
        if len(attachments) >= max_images:
            break
        attachment = downloaded[index]
        handle = f"search_img_{start_index + len(attachments)}"
        basename = os.path.basename(attachment.local_path)
        suffix = basename.split("__", 1)[1] if "__" in basename else basename
        local_path = os.path.join(workdir, f"{handle}__{suffix}")
        if local_path != attachment.local_path:
            os.replace(attachment.local_path, local_path)
        attachments.append(replace(attachment, handle=handle, local_path=local_path))
    kept_paths = {attachment.local_path for attachment in attachments}
    for attachment in downloaded.values():
        if attachment.local_path not in kept_paths and os.path.exists(attachment.local_path):
            os.remove(attachment.local_path)
    return attachments


class MultimodalSearch(BrowserSearch):
    @staticmethod
    def description_for_pixel_feedback(enabled: bool) -> str:
        result_contract = (
            "the matching candidate image pixels as `search_img_N` references so you can "
            "visually compare them with the query image"
            if enabled else
            "no candidate image pixels"
        )
        return (
            "Searches the web. Provide `query` for text search, or `image` (for example "
            "'img_1', a saved crop reference, a URL, or base64) for reverse image search. "
            "Reverse search returns up to 10 source-page text records and "
            f"{result_contract}."
        )

    description = description_for_pixel_feedback(False)
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text search query string."},
            "image": {
                "type": "string",
                "description": "Image reference: 'img_n', a URL, or base64.",
            },
            "topn": {
                "type": "integer",
                "description": "Number of results to display",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.parameters = json.loads(json.dumps(type(self).parameters))
        self.parameters["oneOf"] = [
            {"required": ["query"]},
            {"required": ["image"]},
        ]

    def call(self, params: Union[str, dict], **kwargs) -> str:
        session = kwargs.get("session")
        if not isinstance(session, BrowserSession):
            return "[Tool Error] browser.search requires an active BrowserSession."
        if not isinstance(params, dict):
            return "[Tool Error] browser.search arguments must be an object."
        image_ref = params.get("image") or params.get("image_id") or ""
        query = params.get("query")
        if image_ref and query:
            return "[Tool Error] browser.search accepts exactly one of `query` or `image`, not both."
        if not image_ref:
            return super().call(params, **kwargs)
        reverse_enabled = kwargs.get("reverse_image_search_enabled")
        if reverse_enabled is None:
            reverse_enabled = kwargs.get("web_multimodal_enabled", False)
        if not bool(reverse_enabled):
            return "[Tool Error] browser.search image input is disabled in text mode."
        topn = params.get("topn", 10)
        try:
            from .. import image_search as image_search_backend

            limit = max(1, min(int(topn), 10))
            public_url, records = image_search_backend.search_image_records(
                str(image_ref), limit=limit
            )
            label = params.get("query") or params.get("image_label") or str(image_ref)
            text = session.search_image(records, label=label)
        except (BrowserUsageError, TypeError, ValueError) as exc:
            return str(exc)
        except Exception as exc:
            return f"[browser.search:image] {exc}"
        workdir = str(kwargs.get("mm_search_workdir") or "")
        pixel_feedback = kwargs.get("reverse_image_pixel_feedback_enabled")
        if pixel_feedback is None:
            pixel_feedback = kwargs.get("web_multimodal_enabled", False)
        if not bool(pixel_feedback) or not workdir:
            return text
        attachments = fetch_result_image_attachments(
            records,
            workdir,
            start_index=int(kwargs.get("mm_search_start", 1)),
        )
        return BrowserResult(text, attachments)


__all__ = ["BrowserSearch", "MultimodalSearch", "fetch_result_image_attachments"]
