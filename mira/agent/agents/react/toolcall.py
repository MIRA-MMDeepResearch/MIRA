import json
import re
from typing import Dict, Optional, Tuple

ARG_KEYS = ("query", "q", "url", "goal", "code", "image_id", "bbox")
TOOL_ALIAS = {
    "web_search": "text_search",
    "local_search": "text_search",
    "google_search": "search",
    "functions": "search",
    "functions.search": "search",
    "web_visit": "visit",
}
ARG_KEY_TO_TOOL = {
    "q": "text_search",
    "query": "search",
    "url": "visit",
    "goal": "visit",
    "code": "PythonInterpreter",
    "image_id": "crop_and_search",
    "bbox": "crop_and_search",
}


class ToolCallParsingMixin:
    @staticmethod
    def _parse_xml_tool_call(raw_tool_call: str) -> Optional[Dict]:
        fn = re.search(r'<function\s*=\s*([A-Za-z_][\w\.]*)\s*>', raw_tool_call)
        if not fn:
            return None
        name = fn.group(1)
        args = {}
        for m in re.finditer(
            r'<parameter\s*=\s*(\w+)\s*>(.*?)</parameter>',
            raw_tool_call, re.DOTALL,
        ):
            key = m.group(1)
            val = m.group(2).strip()
            try:
                args[key] = json.loads(val)
            except Exception:
                args[key] = val
        return {"name": name, "arguments": args}

    def _normalize_tool_call(self, raw_tool_call: str) -> Tuple[Optional[str], Optional[Dict], Optional[str]]:
        extracted_code = None
        json_part = raw_tool_call
        code_match = re.search(r'<code>(.*?)</code>', raw_tool_call, re.DOTALL)
        if code_match:
            extracted_code = code_match.group(1).strip()
            json_part = raw_tool_call[:code_match.start()].strip()

        tool_call = self._parse_xml_tool_call(json_part)
        if tool_call is None:
            try:
                tool_call = json.loads(json_part)
            except Exception:
                tool_call = self._recover_tool_call(raw_tool_call, extracted_code)
                if tool_call is None:
                    raw_tool_name = ["__invalid_tool_call_json__", raw_tool_call]
                    return (
                        None,
                        None,
                        f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
                    )

        if isinstance(tool_call, dict):
            raw_tool_name = tool_call.get("name", "")
            if not raw_tool_name and isinstance(tool_call.get("function"), str):
                raw_tool_name = tool_call["function"]
            raw_tool_args = tool_call.get("arguments", {})
        else:
            raw_tool_name = tool_call
            raw_tool_args = {}

        tool_name = raw_tool_name
        if isinstance(tool_name, list):
            if len(tool_name) == 1 and isinstance(tool_name[0], str):
                tool_name = tool_name[0]
            else:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
                )

        if not isinstance(tool_name, str) or not tool_name.strip():
            return (
                None,
                None,
                f"[Tool Error] Invalid tool call format: name must be string, got {raw_tool_name!r}",
            )
        tool_name = tool_name.strip()

        tool_args = raw_tool_args
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool arguments JSON for {tool_name}: {raw_tool_args!r}",
                )
        elif isinstance(tool_args, list):
            if len(tool_args) == 1 and isinstance(tool_args[0], dict):
                tool_args = tool_args[0]
            else:
                return (
                    None,
                    None,
                    f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}",
                )
        elif tool_args is None:
            tool_args = {}

        if not isinstance(tool_args, dict):
            return (
                None,
                None,
                f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {raw_tool_args!r}",
            )

        if extracted_code and tool_name and tool_name.strip() == "PythonInterpreter":
            tool_args["code"] = extracted_code

        return tool_name, tool_args, None

    def _recover_tool_call(self, raw_tool_call: str, extracted_code: Optional[str]) -> Optional[Dict]:
        args = self._recover_tool_args(raw_tool_call)
        if args is None:
            return None
        name = self._recover_tool_name(raw_tool_call)
        if name is None:
            known_tools = self._known_tools()
            if ("image_id" in args or "bbox" in args) and "crop_and_search" in known_tools:
                name = "crop_and_search"
            else:
                for key in args:
                    inferred = ARG_KEY_TO_TOOL.get(key)
                    if inferred and inferred in known_tools:
                        name = inferred
                        break
        if name is None:
            return None
        return {"name": name, "arguments": args}

    def _recover_tool_name(self, raw: str) -> Optional[str]:
        name_pats = (
            r'<?\bfunction_name\s*>\s*["\s]*([A-Za-z_][\w\.]*)',
            r'<?\bfunction\b["\s]*[=:>\s"]+["\s:]*([A-Za-z_][\w\.]*)',
            r'<?\bparameter\b["\s]*[=:>\s"]+["\s:]*([A-Za-z_][\w\.]*)',
            r'<?\bname\b["\s]*[=:>,\s"]+["\s:]*([A-Za-z_][\w\.]*)',
        )
        known = self._known_tools()
        for pat in name_pats:
            m = re.search(pat, raw)
            if not m:
                continue
            tok = m.group(1)
            for cand in (tok, tok.rstrip("."), tok.split(".", 1)[0]):
                nm = TOOL_ALIAS.get(cand, cand)
                if nm in known:
                    return nm
        return None

    @staticmethod
    def _find_balanced_object(s: str, start: int) -> Optional[str]:
        depth = 0
        in_str = False
        esc = False
        i = start
        while i < len(s):
            c = s[i]
            if in_str:
                if esc:
                    esc = False
                elif c == '\\':
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == '{':
                    depth += 1
                elif c == '}':
                    depth -= 1
                    if depth == 0:
                        return s[start:i + 1]
            i += 1
        return None

    def _recover_tool_args(self, raw: str) -> Optional[Dict]:
        known_tools = self._known_tools()

        def _try_json(t):
            try:
                return True, json.loads(t)
            except Exception:
                return False, None

        params = {}
        for m in re.finditer(
            r'<parameter\s*=\s*(\w+)\s*>(.*?)(?:</parameter>|<parameter|</function>|</tool_call>|$)',
            raw, re.DOTALL,
        ):
            key = m.group(1)
            if key in known_tools:
                continue
            val = m.group(2).strip()
            if not val or val == 'invalid JSON?':
                continue
            ok, parsed = _try_json(val)
            params[key] = parsed if ok else val
        if params:
            return params

        am = re.search(r'arguments["\s]*:\s*\{', raw)
        if am:
            obj = self._find_balanced_object(raw, raw.index('{', am.start()))
            if obj:
                ok, parsed = _try_json(obj)
                if ok and isinstance(parsed, dict):
                    return parsed

        for key in ARG_KEYS:
            km = re.search(r'\{\s*"' + key + r'"\s*:', raw)
            if km:
                obj = self._find_balanced_object(raw, km.start())
                if obj:
                    ok, parsed = _try_json(obj)
                    if ok and isinstance(parsed, dict):
                        return parsed

        am2 = re.search(r'\[(?:\s*"(?:[^"\\]|\\.)*"\s*,?)+\]', raw, re.DOTALL)
        if am2:
            ok, parsed = _try_json(am2.group(0))
            if ok:
                return {"query": parsed}

        terms = [
            t for t in re.findall(r'"((?:[^"\\]|\\.)*)"', raw)
            if t.strip() and t not in ARG_KEYS and t not in known_tools
        ]
        if terms:
            return {"query": [" ".join(terms)]}

        return None
