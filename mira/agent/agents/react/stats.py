import json
from contextlib import nullcontext
from typing import Dict, List

from .messages import _message_content_text
from .wire import _extract_tool_call_bodies


class TrajectoryStatsMixin:
    def _active_timing(self):
        return getattr(self._get_thread_state(), "infer_timing", None)

    def _timed_stage(self, name: str):
        trace = self._active_timing()
        return trace.span(name) if trace is not None else nullcontext()

    def _timing_add_stage(self, name: str, seconds: float) -> None:
        trace = self._active_timing()
        if trace is not None:
            trace.add_stage(name, seconds)

    def _build_trajectory_stats(
        self,
        messages: List[Dict],
        timing_snapshot: Dict,
    ) -> Dict:
        metrics = timing_snapshot.get("metrics") or {}
        stages = timing_snapshot.get("stages_s") or {}
        rounds = timing_snapshot.get("rounds") or []

        tool_calls_by_name = {
            name[len("tool."):-len(".calls")]: int(value)
            for name, value in metrics.items()
            if name.startswith("tool.") and name.endswith(".calls")
        }
        tool_errors_by_name = {}
        for row in rounds:
            for tool in row.get("tools") or []:
                if not tool.get("error"):
                    continue
                name = str(tool.get("name") or "unknown")
                tool_errors_by_name[name] = tool_errors_by_name.get(name, 0) + 1

        search_queries = 0
        reverse_image_searches = 0
        visit_urls = 0
        visit_image_opens = 0
        visit_page_image_opens = 0
        visit_question_image_reopens = 0
        for message in messages:
            if message.get("role") != "assistant":
                continue
            parsed_calls = []
            native_calls = message.get("tool_calls") or []
            if native_calls:
                for call in native_calls:
                    function = call.get("function") or {}
                    name = function.get("name")
                    arguments = function.get("arguments") or {}
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            arguments = {}
                    if name and isinstance(arguments, dict):
                        parsed_calls.append((name, arguments))
            else:
                content = _message_content_text(message.get("content", ""))
                for body in _extract_tool_call_bodies(content):
                    name, arguments, error = self._normalize_tool_call(body)
                    if error is None and name and isinstance(arguments, dict):
                        parsed_calls.append((name, arguments))

            for name, arguments in parsed_calls:
                if name == "search":
                    query = arguments.get("query")
                    search_queries += len(query) if isinstance(query, list) else int(bool(query))
                    if arguments.get("image") or arguments.get("image_id"):
                        reverse_image_searches += 1
                elif name == "visit":
                    urls = arguments.get("url")
                    visit_urls += len(urls) if isinstance(urls, list) else int(bool(urls))
                    image_ref = str(
                        arguments.get("image") or arguments.get("image_url") or ""
                    ).strip()
                    if image_ref:
                        visit_image_opens += 1
                        if image_ref.startswith("page_img_") or image_ref.startswith(("http://", "https://")):
                            visit_page_image_opens += 1
                        elif image_ref.startswith("img_"):
                            visit_question_image_reopens += 1

        exposures = list(getattr(self._get_thread_state(), "image_exposures", []) or [])
        handles = []
        for exposure in exposures:
            handles.extend(exposure.get("visible_image_handles") or [])
        message_image_part_count = self._message_image_part_count(messages)
        message_image_handles = self._message_image_part_handles(messages)
        handles.extend(message_image_handles)
        unique_handles = list(dict.fromkeys(handles))
        page_handles = [handle for handle in unique_handles if str(handle).startswith("page_img_")]
        assets = getattr(self._get_thread_state(), "mm_assets", {}) or {}
        known_page_handles = sorted({
            str(record.get("handle") or handle)
            for handle, record in assets.items()
            if str(record.get("handle") or handle).startswith("page_img_")
        })

        stats = {
            "schema_version": 1,
            "llm_calls": int(metrics.get("llm.calls", 0)),
            "tool_calls": {
                "total": sum(tool_calls_by_name.values()),
                "by_name": tool_calls_by_name,
                "errors": sum(tool_errors_by_name.values()),
                "errors_by_name": tool_errors_by_name,
            },
            "tool_inputs": {
                "search_queries": search_queries,
                "reverse_image_searches": reverse_image_searches,
                "visit_urls": visit_urls,
                "visit_image_opens": visit_image_opens,
                "visit_page_image_opens": visit_page_image_opens,
                "visit_question_image_reopens": visit_question_image_reopens,
            },
            "context": {
                "max_prompt_tokens": int(metrics.get("context.max_tokens", 0)),
                "management_triggers": int(metrics.get("context.management_triggers", 0)),
                "management_strategy": str(self.memory_strategy),
                "threshold_tokens": int(self.base_memory_context_threshold),
            },
            "images": {
                "input_images": len(getattr(self._get_thread_state(), "image_refs", []) or []),
                "exposure_events": len(exposures),
                "one_shot_pixel_exposures": sum(
                    int(row.get("visible_image_count") or 0) for row in exposures
                ),
                "message_image_parts": message_image_part_count,
                "message_image_handles": message_image_handles,
                "pixel_exposures": (
                    sum(int(row.get("visible_image_count") or 0) for row in exposures)
                    + message_image_part_count
                ),
                "unique_visible_images": len(unique_handles),
                "unique_visible_handles": unique_handles,
                "webpage_images": len(page_handles),
                "webpage_image_pixel_exposures": len(page_handles),
                "known_webpage_image_handles": known_page_handles,
                "known_webpage_image_markers": len(known_page_handles),
            },
            "wall_time_s": round(float(stages.get("agent.total", 0.0)), 6),
        }
        return stats

    def _timing_increment(self, name: str, amount: int = 1) -> None:
        trace = self._active_timing()
        if trace is not None:
            trace.increment(name, amount)

    def _timing_note(self, name: str, value) -> None:
        trace = self._active_timing()
        if trace is not None:
            trace.note(name, value)

    def _emit_round_timing(self, task_id, rollout, row: dict) -> None:
        if self.round_timing_callback is None:
            return
        try:
            self.round_timing_callback({"task_id": task_id, "rollout": rollout or 1, **row})
        except Exception as exc:
            print(f"[infer-timer] failed to persist round timing: {exc}")
