import os
import re
from typing import Dict, List, Optional

from .messages import _message_content_text


class ImageMessageMixin:
    @staticmethod
    def _message_image_part_count(messages: List[Dict]) -> int:
        return sum(
            1
            for message in messages
            for part in (
                message.get("content", [])
                if isinstance(message.get("content"), list)
                else []
            )
            if isinstance(part, dict)
            and part.get("type") in {"image_url", "input_image", "image"}
        )

    def _message_image_part_handles(self, messages: List[Dict]) -> List[str]:
        handles = []
        fallback_input_index = 1
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            image_count = sum(
                1
                for part in content
                if isinstance(part, dict)
                and part.get("type") in {"image_url", "input_image", "image"}
            )
            if not image_count:
                continue
            text = _message_content_text(content)
            labelled = re.findall(r"\[([^\]\n]+)\]<image>", text)
            if labelled:
                handles.extend(labelled[:image_count])
            for _ in range(max(0, image_count - len(labelled))):
                handles.append(f"img_{fallback_input_index}")
                fallback_input_index += 1
        return handles

    def _consume_one_shot_images(
        self,
        messages: List[Dict],
        *,
        actor_call_id: int,
        through_index: Optional[int] = None,
        record: bool = True,
    ) -> Optional[Dict]:
        if self.image_history_policy != "one_shot":
            return None
        limit = len(messages) - 1 if through_index is None else int(through_index)
        removed = 0
        handles = []
        unlabelled_messages = []
        for index, message in enumerate(messages):
            if index > limit:
                break
            content = message.get("content")
            if not isinstance(content, list):
                continue
            image_count = sum(
                1
                for part in content
                if isinstance(part, dict)
                and part.get("type") in {"image_url", "input_image", "image"}
            )
            if not image_count:
                continue
            text = _message_content_text(content)
            labelled = re.findall(r"\[([^\]\n]+)\]<image>", text)
            handles.extend(labelled[:image_count])
            if len(labelled) < image_count:
                unlabelled_messages.append((index, image_count - len(labelled)))

            stripped = []
            for part in content:
                if (
                    isinstance(part, dict)
                    and part.get("type") in {"image_url", "input_image", "image"}
                ):
                    removed += 1
                    continue
                if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                    key = "text" if part.get("type") == "text" else "input_text"
                    value = str(part.get(key) or "")
                    value = re.sub(
                        r"(\[[^\]\n]+\])<image>",
                        r"\1<image_omitted>",
                        value,
                    )
                    part = dict(part, **{key: value})
                stripped.append(part)
            message["content"] = stripped

        if not removed:
            return None

        next_input = 1
        for message_index, missing in unlabelled_messages:
            inferred = []
            for _ in range(missing):
                handle = f"img_{next_input}"
                next_input += 1
                inferred.append(handle)
                handles.append(handle)
            marker = (
                "[Earlier image pixels were shown once and are omitted from later "
                "actor context. These handles remain only as tool references, not "
                f"as visible pixels in the current context: {', '.join(inferred)}.]"
            )
            content = messages[message_index].get("content")
            if isinstance(content, list):
                content.append({"type": "text", "text": marker})

        exposure = {
            "actor_call_id": int(actor_call_id),
            "visible_image_count": removed,
            "visible_image_handles": handles,
        }
        if record:
            self._get_thread_state().image_exposures.append(exposure)
        print(
            "[image-history] one-shot consumed "
            f"actor_call={actor_call_id} images={removed} handles={handles}"
        )
        return exposure

    def _prepare_one_shot_resume(self, messages: List[Dict]) -> None:
        if self.image_history_policy != "one_shot":
            return
        last_assistant = max(
            (index for index, msg in enumerate(messages) if msg.get("role") == "assistant"),
            default=-1,
        )
        if last_assistant >= 0:
            self._consume_one_shot_images(
                messages,
                actor_call_id=-1,
                through_index=last_assistant,
                record=False,
            )

    def _item_image_refs(self, item) -> List[str]:
        images = item.get("images") or item.get("image_url") or item.get("image") or []
        if isinstance(images, str):
            images = [images]
        refs = []
        for image in images:
            image = str(image).strip()
            if image:
                refs.append(image)
        return refs

    def _openai_image_url(self, image_ref: str) -> str:
        if image_ref.startswith(("http://", "https://", "file://", "data:image/")):
            return image_ref
        return f"file://{os.path.abspath(image_ref)}"


    def _mm_protocol_user_message(
        self, question: str, image_refs: List[str]
    ) -> Dict:
        clean_question = str(question).strip()
        text = f"Question: {clean_question}"
        content: List[Dict] = [{"type": "text", "text": text}]
        for image_ref in image_refs:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._openai_image_url(image_ref)},
            })
        return {"role": "user", "content": content}
