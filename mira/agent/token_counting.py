from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Mapping






class JsonVocabTokenCounter:
    per_message_overhead = 4

    def __init__(self, model_path: str):
        self.model_path = str(model_path or "")
        vocab_file = os.path.join(self.model_path, "tokenizer.json")
        if not os.path.isfile(vocab_file):
            raise FileNotFoundError(f"tokenizer.json not found: {vocab_file}")

        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(vocab_file)

    def encode(self, text, allow_special: bool = False):
        return self.tokenizer.encode(str(text), add_special_tokens=False).ids

    def count_messages(self, messages, tools=None) -> tuple[int, int, int]:
        chunks = []
        for message in messages or []:
            if not isinstance(message, Mapping):
                continue
            chunks.append(str(message.get("role") or ""))
            content = message.get("content")
            if isinstance(content, list):
                chunks.extend(
                    str(part.get("text") or "")
                    for part in content
                    if isinstance(part, Mapping)
                )
            elif content is not None:
                chunks.append(str(content))
            for key in ("reasoning", "tool_calls", "name", "tool_call_id"):
                value = message.get(key)
                if value:
                    chunks.append(
                        value if isinstance(value, str)
                        else json.dumps(value, ensure_ascii=False)
                    )
        if tools:
            chunks.append(json.dumps(tools, ensure_ascii=False))
        text_tokens = len(self.encode("\n".join(chunks)))
        text_tokens += self.per_message_overhead * len(messages or [])
        return text_tokens, text_tokens, 0


@lru_cache(maxsize=8)
def get_model_token_counter(model_path: str):
    path = str(Path(model_path).expanduser())
    return JsonVocabTokenCounter(path)




__all__ = ["JsonVocabTokenCounter", "get_model_token_counter"]
