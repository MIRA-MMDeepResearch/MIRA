import copy
import json
import re
from typing import Dict, List, Optional

from ...prompts import PROMPT_REGISTRY
from .messages import _tool_result_texts, _render_tool_responses


class PromptSchemaMixin:
    @staticmethod
    def _function_entry_name(entry) -> Optional[str]:
        if isinstance(entry, str):
            return entry
        if isinstance(entry, dict):
            if isinstance(entry.get("name"), str):
                return entry["name"]
            function = entry.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                return function["name"]
            return None
        name = getattr(entry, "name", None)
        return name if isinstance(name, str) else None

    def build_system_prompt(self, enable_python_tool: bool) -> str:
        prompt_name = self.cfg.prompt.system_prompt_name
        prompt = PROMPT_REGISTRY.get(prompt_name)
        if prompt is None:
            raise SystemExit(f"[build_system_prompt] unknown system_prompt_name {prompt_name!r}; known: {sorted(PROMPT_REGISTRY)}")
        from ...prompts import SELF_CONTAINED_PROMPTS
        if prompt_name in SELF_CONTAINED_PROMPTS:
            return prompt
        return prompt

    def tool_result_role(self) -> str:
        cfg = getattr(self, "cfg", None)
        role = (cfg.model.tool_result_role or "") if cfg is not None else ""
        return role.strip() or self.default_tool_result_role

    def _pending_tool_call_ids(self):
        tl = self._get_thread_state()
        native = getattr(tl, "native_assistant_message", None) or {}
        return [call.get("id") for call in native.get("tool_calls") or []]

    def build_tool_response_messages(self, result):
        texts = _tool_result_texts(result)
        if self.tool_result_role() == "tool":
            ids = self._pending_tool_call_ids()
            if len(ids) == len(texts) and all(ids):
                return [
                    {"role": "tool", "tool_call_id": call_id, "content": text}
                    for call_id, text in zip(ids, texts)
                ]
        return [{"role": "user", "content": _render_tool_responses(result)}]

    def drain_tool_call_timings(self):
        return []

    def build_assistant_response_message(self, content):
        native = getattr(self._get_thread_state(), "native_assistant_message", None)
        if native:
            return native
        return {"role": "assistant", "content": content.strip()}

    default_tool_schema_channel = "openai_tools"

    default_tool_result_role = "user"

    def api_tools(self):
        return None

    def _openai_tool_specs(self):
        specs = []
        for entry in self.function_list:
            name = self._function_entry_name(entry)
            tool = self._tool_map().get(name)
            if tool is None:
                continue
            specs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            })
        return specs

    def tool_schema_channel(self) -> str:
        cfg = getattr(self, "cfg", None)
        explicit = ((cfg.model.tool_schema_channel or "") if cfg is not None else "").strip()
        if explicit:
            return explicit
        from ...prompts import SELF_CONTAINED_PROMPTS
        name = cfg.prompt.system_prompt_name if cfg is not None else ""
        if name in SELF_CONTAINED_PROMPTS:
            return "system_prompt"
        return self.default_tool_schema_channel

    def _pinned_tool_schema(self):
        cfg = getattr(self, "cfg", None)
        path = ((cfg.model.tool_schema_file or "") if cfg is not None else "").strip()
        if not path:
            return None
        cached = getattr(self, "_pinned_tool_schema_cache", None)
        if cached is None or cached[0] != path:
            with open(path, encoding="utf-8") as handle:
                cached = (path, json.load(handle))
            self._pinned_tool_schema_cache = cached
        return cached[1]

    def delivered_api_tools(self):
        if self.tool_schema_channel() == "system_prompt":
            return None
        specs = self.api_tools() or self._openai_tool_specs()
        pinned = self._pinned_tool_schema()
        if pinned is None:
            return specs
        live = {(spec.get("function") or {}).get("name") for spec in specs}
        frozen = {(spec.get("function") or {}).get("name") for spec in pinned}
        if live != frozen:
            raise ValueError(
                "model.tool_schema_file declares tools the run does not enable: "
                f"pinned={sorted(frozen)} enabled={sorted(live)}"
            )
        return copy.deepcopy(pinned)

    def wire_tool_name_maps(self):
        real_to_alias, alias_to_real = {}, {}
        for spec in self.delivered_api_tools() or []:
            name = ((spec.get("function") or {}).get("name") or "")
            alias = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
            if alias and alias != name:
                real_to_alias[name] = alias
                alias_to_real[alias] = name
        return real_to_alias, alias_to_real

    def _delivered_tool_schema_for_audit(self) -> List[Dict]:
        return copy.deepcopy(self.delivered_api_tools() or [])

    @staticmethod
    def _render_tool_schema_text_for_audit(tools: List[Dict]) -> str:
        if not tools:
            return ""
        return (
            "\n\n<delivered_tool_schema>\n"
            + json.dumps(tools, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n</delivered_tool_schema>"
        )

    def _system_prompt_with_tool_schema_for_audit(self, messages: List[Dict]) -> str:
        system_prompt = ""
        for message in messages or []:
            if message.get("role") != "system":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                system_prompt = "\n".join(
                    str(part.get("text") or part.get("input_text") or "")
                    for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"text", "input_text"}
                )
            else:
                system_prompt = str(content or "")
            break
        tools = self._delivered_tool_schema_for_audit()
        return system_prompt + self._render_tool_schema_text_for_audit(tools)

    def _apply_tool_schema_channel(self, prompt: str) -> str:
        return prompt
