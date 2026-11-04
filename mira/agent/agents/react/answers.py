import copy
import re
from typing import Dict, List, Optional

from .wire import _strip_thinking


class AnswerExtractionMixin:

    @staticmethod
    def _is_pseudo_answer(candidate: str) -> bool:
        s = candidate.strip()
        if re.search(
            r"<\s*/?\s*think\b|<\|(?:open|close)\|>|<\|sep\|>",
            s,
            flags=re.IGNORECASE,
        ):
            return True
        if len(s) >= 500:
            return False
        low = s.lower()
        INTENT = (
            "let me", "let's", "i'll", "i will", "i am going to", "i'm going to",
            "i need to", "i should", "i want to", "next step", "next, ",
            "\u8ba9\u6211", "\u6211\u5c06", "\u6211\u4f1a", "\u6211\u5148", "\u6211\u73b0\u5728", "\u6211\u9700\u8981", "\u63a5\u4e0b\u6765", "\u4e0b\u4e00\u6b65", "\u7136\u540e\u6211",
        )
        if any(p in low for p in INTENT):
            return True
        if s.endswith(":") or s.endswith("\uff1a"):
            return True
        return False

    @staticmethod
    def _extract_tagged_answer(content: str) -> str:
        text = _strip_thinking(content)
        valid = []
        for match in re.finditer(
                r"(?=<answer>(.*?)</answer>)", text,
                flags=re.DOTALL | re.IGNORECASE):
            candidate = match.group(1).strip()
            if not candidate:
                continue
            if AnswerExtractionMixin._is_pseudo_answer(candidate):
                continue
            valid.append(candidate)
        return valid[-1] if valid else ""

    @staticmethod
    def _has_tool_intent(text: str) -> bool:
        t = str(text or "")
        t = _strip_thinking(t)
        return bool(re.search(
            r"<tool_call>|</tool_call>|<function=[\w.]|</?\uFF5CDSML\uFF5C|<parameter=\w", t))

    @staticmethod
    def _strip_tool_markup(text: str) -> str:
        text = re.sub(r"<tool_call>.*?(?:</tool_call>|$)", "", text,
                      flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<\uFF5CDSML\uFF5Ctool_calls>.*?(?:</\uFF5CDSML\uFF5Ctool_calls>|$)",
                      "", text, flags=re.DOTALL)
        text = re.sub(r"<\uFF5CDSML\uFF5C[^>]*>", "", text)
        text = re.sub(r"<function=[\w.]+>.*?(?:</function>|$)", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_answer_candidate(content: str) -> str:
        text = str(content or "").strip()
        if not text or text.lower() in {"answer not found.", "no answer found."}:
            return ""
        if "vllm server error" in text.lower():
            return ""
        tagged = AnswerExtractionMixin._extract_tagged_answer(text)
        if tagged:
            return tagged
        cleaned = _strip_thinking(text).strip()
        match = re.search(r"\\boxed\{([^{}]+)\}", cleaned)
        if match:
            return match.group(1).strip()
        cleaned = AnswerExtractionMixin._strip_tool_markup(cleaned)
        cleaned = re.sub(r"</?(?:answer|think)>", "", cleaned,
                         flags=re.IGNORECASE).strip()
        if not cleaned:
            return ""
        return cleaned[:16000].strip()


    @staticmethod
    def _is_degenerate_repetition(content: str, *, n: int = 30,
                                  threshold: float = 0.5,
                                  min_words: int = 1200) -> bool:
        text = str(content or "")
        if len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text)) <= min_words:
            return False
        if len(text) < n:
            return False
        ngrams = [text[i:i + n] for i in range(len(text) - n + 1)]
        return bool(ngrams) and len(set(ngrams)) / len(ngrams) < threshold

    def _compact_final_answer_retry(self, question: str, messages: List[Dict], tl,
                                    reason: str, *, timeout_override_s: float = 300,
                                    deadline_monotonic: Optional[float] = None) -> str:
        try:
            finalizer_user = (
                "Research is over. Tools are disabled.\n\n"
                "Give your best final answer now.\n"
                "Put only the final answer text between <answer> and </answer>."
            )
            retry_record = {
                "role": "user",
                "content": finalizer_user,
                "finalization_retry": True,
            }
            messages.append(retry_record)
            tl.full_messages.append(copy.deepcopy(retry_record))
            retry_content = self.call_server(
                messages,
                max_tries=2,
                disable_tools=True,
                timeout_override_s=timeout_override_s,
                deadline_monotonic=deadline_monotonic,
            )
            retry_assistant = {
                "role": "assistant",
                "content": str(retry_content or "").strip(),
                "finalization_retry": True,
            }
            messages.append(retry_assistant)
            tl.full_messages.append(copy.deepcopy(retry_assistant))
            return str(retry_content or "")
        except Exception as e:
            print(f"[Finalization] final-answer retry failed: {e}")
            return ""
