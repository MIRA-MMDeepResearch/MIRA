import re


def _normalize_tool_markup(content):
    content = re.sub(
        r"<\s*(/?)\s*tool_call\s*>",
        lambda match: f"<{match.group(1)}tool_call>",
        str(content or ""), flags=re.IGNORECASE,
    )
    return re.sub(r"</tool_call>\s*(?=<code>)", "", content, flags=re.IGNORECASE)


_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>|"
    r"<\|begin_of_thought\|>.*?<\|end_of_thought\|>",
    flags=re.DOTALL | re.IGNORECASE,
)


def _strip_thinking(content):
    text = _THINK_BLOCK_RE.sub("", str(content or ""))
    close = re.search(r"</think(?:ing)?>", text, flags=re.IGNORECASE)
    if close and not re.search(r"<think(?:ing)?>", text[:close.start()], flags=re.IGNORECASE):
        text = text[close.end():]
    return text

def capture_native_assistant(tl, message, content):
    calls = []
    for call in (getattr(message, "tool_calls", None) or []):
        function = getattr(call, "function", None)
        calls.append({
            "id": getattr(call, "id", None),
            "type": getattr(call, "type", None) or "function",
            "function": {
                "name": getattr(function, "name", None),
                "arguments": getattr(function, "arguments", None) or "{}",
            },
        })
    if not calls:
        tl.native_assistant_message = None
        return []
    native = {"role": "assistant", "content": content, "tool_calls": calls}
    for field in ("reasoning_content", "reasoning"):
        value = getattr(message, field, None)
        if value:
            native[field] = value
            break
    tl.native_assistant_message = native
    return calls


def _extract_tool_call_bodies(content):
    text = str(content or "")
    bodies = []
    start = 0
    for think in _THINK_BLOCK_RE.finditer(text):
        bodies.extend(re.findall(
            r"<tool_call>(.*?)</tool_call>",
            text[start:think.start()], re.DOTALL,
        ))
        bodies.extend(re.findall(
            r"<tool_call>(.*?)</tool_call>",
            think.group(0), re.DOTALL,
        ))
        start = think.end()
    bodies.extend(re.findall(
        r"<tool_call>(.*?)</tool_call>", text[start:], re.DOTALL,
    ))
    return bodies
