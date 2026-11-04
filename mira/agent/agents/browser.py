import copy
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Tuple

from .base import (
    BaseReactAgent,
    ToolResultBatch,
    _extract_tool_call_bodies,
)
from .mm_mixin import MMToolMixin
from ..tools.browser.search import BrowserSearch, MultimodalSearch
from ..tools.browser.session import BrowserSession
from ..tools.browser.find import Find
from ..tools.browser.open import (
    DESCRIPTION_TEXT_ONLY as OPEN_DESCRIPTION_TEXT_ONLY,
    ID_DESCRIPTION_TEXT_ONLY as OPEN_ID_DESCRIPTION_TEXT_ONLY,
    Open,
    image_handle as _open_image_handle,
)


_BROWSER_TOOL_CLASSES = {
    BrowserSearch.name: BrowserSearch,
    Open.name: Open,
    Find.name: Find,
}


class BrowserReactAgent(MMToolMixin, BaseReactAgent):
    default_function_list = ["browser.search", "browser.open", "browser.find"]

    def _browser_feature(self, name: str, *, legacy_page: bool = False) -> bool:
        value = getattr(self, name, None)
        if value is not None:
            return bool(value)
        if legacy_page:
            value = getattr(self, "_web_page_images_enabled", None)
            if value is not None:
                return bool(value)
        value = getattr(self, "_web_multimodal_enabled", None)
        if value is not None:
            return bool(value)
        return bool(getattr(self, "_mm_protocol", False))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requested = {
            name
            for entry in self.function_list
            if (name := self._function_entry_name(entry))
        }
        self._enabled_tool_names = {
            self._canonical_browser_name(name) or name for name in requested
        }
        self._web_multimodal_enabled = bool(
            self.reverse_image_search_enabled
            or self.page_image_pixel_feedback_enabled
            or self.page_image_open_enabled
            or self.python_image_auto_feedback_enabled
        )
        self._mm_protocol = self.image_workspace_enabled
        self._web_page_images_enabled = self.page_image_pixel_feedback_enabled
        tool_classes = dict(_BROWSER_TOOL_CLASSES)
        if self.reverse_image_search_enabled:
            tool_classes["browser.search"] = MultimodalSearch
        self._browser_tools = {
            name: tool_class()
            for name, tool_class in tool_classes.items()
            if name in self._enabled_tool_names
        }
        reverse_search_tool = self._browser_tools.get("browser.search")
        if isinstance(reverse_search_tool, MultimodalSearch):
            reverse_search_tool.description = MultimodalSearch.description_for_pixel_feedback(
                self.reverse_image_pixel_feedback_enabled
            )
        if not self.page_image_open_enabled and "browser.open" in self._browser_tools:
            tool = self._browser_tools["browser.open"]
            tool.parameters = copy.deepcopy(tool.parameters)
            tool.parameters["properties"].pop("image", None)
            tool.parameters["properties"]["id"]["description"] = OPEN_ID_DESCRIPTION_TEXT_ONLY
            tool.description = OPEN_DESCRIPTION_TEXT_ONLY

    def _init_thread_state(self):
        super()._init_thread_state()
        self._thread_local.browser_session = BrowserSession()
        self._thread_local.pending_browser_calls = []
        self._thread_local.browser_python_memo = {}
        self._init_mm_thread_state()


    @staticmethod
    def _canonical_browser_name(tool_name: str) -> Optional[str]:
        aliases = {
            "browser.search": "browser.search",
            "browser_search": "browser.search",
            "search": "browser.search",
            "browser.open": "browser.open",
            "browser_open": "browser.open",
            "open": "browser.open",
            "browser.find": "browser.find",
            "browser_find": "browser.find",
            "find": "browser.find",
        }
        return aliases.get(str(tool_name or "").strip().lower())


    def api_tools(self):
        from .base import TOOL_MAP

        tools = []
        for entry in self.function_list:
            name = self._function_entry_name(entry)
            canonical = self._canonical_browser_name(name)
            tool = self._browser_tools.get(canonical) if canonical else None
            if tool is None:
                tool = (TOOL_MAP or {}).get(name)
            if tool is None:
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return tools

    def build_tool_response_messages(self, result):
        return self._attach_mm_images(super().build_tool_response_messages(result), result)

    def _normalize_tool_call(
        self, raw_tool_call: str
    ) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        name, arguments, error = super()._normalize_tool_call(raw_tool_call)
        if error is not None:
            return name, arguments, error
        canonical = self._canonical_browser_name(name)
        if canonical is None:
            return name, arguments, None
        return canonical, arguments, None

    def _remember_pending_browser_calls(self, content):
        calls = _extract_tool_call_bodies(content)
        self._get_thread_state().pending_browser_calls = calls[:-1] if len(calls) > 1 else []
        return content

    def _postprocess_response(self, message, content):
        return self._remember_pending_browser_calls(
            super()._postprocess_response(message, content)
        )

    @staticmethod
    def _available_page_images(assets: dict, limit: int = 12, exclude: dict = None) -> str:
        seen, listed = set(), []
        for handle, asset in (assets or {}).items():
            if not isinstance(asset, dict) or str(asset.get("kind") or "") != "page_image":
                continue
            name = str(asset.get("handle") or handle)
            if name.casefold() in seen or name.casefold() in (exclude or {}):
                continue
            seen.add(name.casefold())
            caption = str(asset.get("caption") or asset.get("alt") or "").strip()
            listed.append(f"{name} ({caption[:40]})" if caption else name)
        if not listed:
            return "No webpage images have been offered yet; open a page first."
        shown = ", ".join(listed[:limit])
        more = f" (+{len(listed) - limit} more)" if len(listed) > limit else ""
        return f"Available now: {shown}{more}."

    def _call_one_tool(self, tool_name: str, tool_args: dict, **kwargs):
        canonical = self._canonical_browser_name(tool_name)
        effective_name = canonical or tool_name
        if effective_name not in self._enabled_tool_names:
            return (
                f"[Tool Error] Unknown tool: {tool_name}. Available tools: "
                f"{sorted(self._enabled_tool_names)}"
            )
        image_workspace = self._browser_feature("image_workspace_enabled")
        reverse_search = self._browser_feature("reverse_image_search_enabled")
        reverse_pixels = self._browser_feature("reverse_image_pixel_feedback_enabled")
        page_pixels = self._browser_feature(
            "page_image_pixel_feedback_enabled", legacy_page=True
        )
        page_open = self._browser_feature("page_image_open_enabled", legacy_page=True)
        if image_workspace and effective_name == "PythonInterpreter":
            registry, workdir = self._ensure_mm_registry()
            kwargs = dict(kwargs, mm_images=registry, mm_workdir=workdir)
        elif effective_name == "PythonInterpreter":
            memo = getattr(self._get_thread_state(), "browser_python_memo", None) or {}
            hit = memo.get(("python", tool_args.get("code")))
            if hit is not None:
                result, exc = hit
                if exc is not None:
                    raise exc
                return result
        if canonical is None:
            return super().custom_call_tool(tool_name, tool_args, **kwargs)
        if reverse_search and canonical == "browser.search":
            tool_args, extra, err = self._resolve_mm_image_arg(tool_args, "browser.search")
            if err is not None:
                return err
            if extra:
                kwargs = dict(kwargs, **extra)
        image_ref = ""
        if page_open and canonical == "browser.open":
            image_ref = _open_image_handle(tool_args.get("id")) or str(
                tool_args.get("image") or tool_args.get("image_url") or ""
            ).strip()
            if image_ref:
                _registry, workdir = self._ensure_mm_registry()
                tl_state = self._get_thread_state()
                assets = getattr(tl_state, "mm_assets", {}) or {}
                failed = getattr(tl_state, "mm_failed_handles", None)
                if failed is None:
                    failed = tl_state.mm_failed_handles = {}
                previous = failed.get(image_ref.casefold())
                if previous:
                    return (
                        f"[Tool Error] browser.open: {image_ref} could not be fetched "
                        f"earlier ({previous}) and will not become available. "
                        f"{self._available_page_images(assets, exclude=failed)}"
                    )
                asset = assets.get(image_ref) or assets.get(image_ref.casefold())
                if not asset or str(asset.get("kind") or "") != "page_image":
                    return (
                        f"[Tool Error] browser.open: {image_ref!r} is not a webpage image handle. "
                        f"{self._available_page_images(assets, exclude=failed)}"
                    )
                tool_args = {k: v for k, v in tool_args.items() if k != "id"}
                source_url = str(asset.get("source_url") or "").strip()
                if not source_url:
                    return (
                        f"[Tool Error] browser.open: image reference {image_ref!r} "
                        "has no hidden source URL in the page-image registry."
                    )
                tool_args = dict(
                    tool_args,
                    image=source_url,
                    image_handle=str(asset.get("handle") or image_ref),
                    image_caption=str(asset.get("caption") or asset.get("alt") or ""),
                    image_page_url=str(asset.get("page_url") or ""),
                )
                kwargs = dict(kwargs, mm_search_workdir=workdir)
        needs_registry = (
            (canonical == "browser.search" and reverse_search)
            or (canonical in {"browser.open", "browser.find"}
                and (page_pixels or page_open))
        )
        if needs_registry:
            _, workdir = self._ensure_mm_registry()
            tl = self._get_thread_state()
            if canonical == "browser.search":
                feedback_start = tl.mm_search_count + 1
                tool_multimodal = reverse_pixels
            else:
                feedback_start = getattr(tl, "mm_page_count", 0) + 1
                tool_multimodal = page_pixels
            kwargs = dict(
                kwargs,
                web_multimodal_enabled=tool_multimodal,
                reverse_image_search_enabled=reverse_search,
                reverse_image_pixel_feedback_enabled=(
                    reverse_pixels
                ),
                page_image_open_enabled=page_open,
                mm_search_workdir=(
                    workdir
                    if canonical != "browser.search"
                    or reverse_pixels
                    else ""
                ),
                mm_search_start=feedback_start,
            )
        session = self._get_thread_state().browser_session
        result = self._browser_tools[canonical].call(tool_args, session=session, **kwargs)
        if page_open and canonical == "browser.open" and image_ref:
            attached = bool(getattr(result, "new_image_paths", None)
                            or getattr(result, "attachments", None))
            if not attached:
                tl_state = self._get_thread_state()
                tl_state.mm_failed_handles[image_ref.casefold()] = str(result)[:70].strip()
        if needs_registry:
            tl = self._get_thread_state()
            image_count = len(getattr(result, "new_image_paths", None) or [])
            if canonical == "browser.search":
                tl.mm_search_count += image_count
            else:
                if page_open:
                    self._register_mm_attachment_metadata(result)
                else:
                    tl.mm_page_count = getattr(tl, "mm_page_count", 0) + image_count
        return result

    def _prefetch_browser_batch(self, calls, session):
        python_parallel = not self._browser_feature("image_workspace_enabled")
        search_jobs, fetch_jobs, python_jobs = [], [], []
        for name, arguments, error in calls:
            if error is not None or not isinstance(arguments, dict):
                continue
            canonical = self._canonical_browser_name(name)
            if canonical == "browser.search" and not (
                arguments.get("image") or arguments.get("image_id")
            ):
                try:
                    topn = max(1, min(int(arguments.get("topn", 10) or 10), 10))
                except (TypeError, ValueError):
                    topn = 10
                query = arguments.get("query")
                for item in query if isinstance(query, list) else [query]:
                    if isinstance(item, str) and item.strip():
                        search_jobs.append((item, topn))
            elif canonical == "browser.open":
                target = arguments.get("id")
                if (
                    isinstance(target, str)
                    and re.match(r"^\s*https?://", target, re.I)
                    and not _open_image_handle(target)
                    and not arguments.get("view_source")
                    and target not in session.pages_by_url
                ):
                    fetch_jobs.append(target)
            elif name == "PythonInterpreter" and python_parallel:
                code = arguments.get("code")
                if isinstance(code, str) and code.strip():
                    python_jobs.append(code)
        search_jobs = list(dict.fromkeys(search_jobs))
        fetch_jobs = list(dict.fromkeys(fetch_jobs))
        python_jobs = list(dict.fromkeys(python_jobs))
        total_jobs = len(search_jobs) + len(fetch_jobs) + len(python_jobs)
        if total_jobs < 2:
            return None
        orig_search, orig_fetch = session.search_fn, session.fetch_fn
        memo = {}

        def warm(kind, key, fn, fn_args):
            try:
                memo[(kind, key)] = (fn(*fn_args), None)
            except Exception as exc:
                memo[(kind, key)] = (None, exc)

        deadline = getattr(self._get_thread_state(), "task_deadline", None)

        def run_python(code):
            extra = {} if deadline is None else {"task_deadline_monotonic": deadline}
            return BaseReactAgent.custom_call_tool(
                self, "PythonInterpreter", {"code": code}, **extra
            )

        with ThreadPoolExecutor(max_workers=min(total_jobs, 8)) as pool:
            for query, topn in search_jobs:
                pool.submit(warm, "search", (query, topn), orig_search, (query, topn))
            for url in fetch_jobs:
                pool.submit(warm, "fetch", url, orig_fetch, (url, False))
            for code in python_jobs:
                pool.submit(warm, "python", code, run_python, (code,))
        self._get_thread_state().browser_python_memo = memo

        def memo_search(query, topn):
            hit = memo.get(("search", (query, topn)))
            if hit is None:
                return orig_search(query, topn)
            result, exc = hit
            if exc is not None:
                raise exc
            return result

        def memo_fetch(url, view_source=False):
            hit = None if view_source else memo.get(("fetch", url))
            if hit is None:
                return orig_fetch(url, view_source)
            result, exc = hit
            if exc is not None:
                raise exc
            return result

        session.search_fn = memo_search
        session.fetch_fn = memo_fetch
        return orig_search, orig_fetch

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        tl = self._get_thread_state()
        pending = list(tl.pending_browser_calls)
        tl.pending_browser_calls = []
        calls = [self._normalize_tool_call(raw_call.strip()) for raw_call in pending]
        calls.append((tool_name, tool_args, None))
        session = tl.browser_session
        restore = self._prefetch_browser_batch(calls, session) if len(calls) > 1 else None
        try:
            results = []
            new_image_paths = []
            attachments = []
            for name, arguments, error in calls[:-1]:
                if error is not None:
                    results.append(error)
                else:
                    one = self._call_one_tool(name, arguments, **kwargs)
                    new_image_paths.extend(getattr(one, "new_image_paths", None) or [])
                    attachments.extend(getattr(one, "attachments", None) or [])
                    results.append(str(one))
            last = self._call_one_tool(tool_name, tool_args, **kwargs)
        finally:
            self._get_thread_state().browser_python_memo = {}
            if restore is not None:
                session.search_fn, session.fetch_fn = restore
        new_image_paths.extend(getattr(last, "new_image_paths", None) or [])
        attachments.extend(getattr(last, "attachments", None) or [])
        results.append(str(last))
        batch = ToolResultBatch(results)
        batch.new_image_paths = new_image_paths
        batch.attachments = attachments
        return batch


__all__ = ["BrowserReactAgent"]
