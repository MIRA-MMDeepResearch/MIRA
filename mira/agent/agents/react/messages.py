import re


OBS_START = '<tool_response>'
OBS_END = '\n</tool_response>'

class ToolResultBatch:
    def __init__(self, results):
        self.results = tuple(str(result) for result in results)


def _tool_result_texts(result):
    if isinstance(result, ToolResultBatch):
        return result.results
    return (str(result),)


def _render_tool_responses(result):
    return "\n".join(
        OBS_START + "\n" + text + OBS_END
        for text in _tool_result_texts(result)
    )


def _message_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content or "")
    texts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"text", "input_text"}:
            value = part.get("text")
            if value is None:
                value = part.get("input_text")
            if value is not None:
                texts.append(str(value))
    return "\n".join(texts)


def _is_tool_response_message(message):
    role = message.get("role")
    content = _message_content_text(message.get("content", ""))
    labelled_observation = re.match(
        r"^(?:\[[^\]\n]+\](?:<image>|<image_omitted>))*<tool_response>", content
    )
    return role == "tool" or (
        role == "user"
        and bool(labelled_observation)
        and content.rstrip().endswith("</tool_response>")
    )
