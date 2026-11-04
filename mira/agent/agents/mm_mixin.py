from __future__ import annotations

import base64
import json
import os
import re
import shutil
import tempfile
from typing import Dict, Optional


class MMToolMixin:
    def _init_mm_thread_state(self):
        self._thread_local.mm_registry = None
        self._thread_local.mm_workdir = None
        self._thread_local.mm_key = None
        self._thread_local.mm_search_count = 0
        self._thread_local.mm_page_count = 0
        self._thread_local.mm_seen_perceptual_hashes = []
        self._thread_local.mm_assets = {}
        self._thread_local.mm_failed_handles = {}

    def _web_mm_active(self) -> bool:
        explicit = getattr(self, "tool_image_feedback_enabled", None)
        if explicit is not None:
            return bool(explicit)
        explicit = getattr(self, "web_multimodal_enabled", None)
        if explicit is not None:
            return bool(explicit)
        return bool(getattr(self, "_mm_protocol", False))


    @staticmethod
    def _materialize_image_ref(ref: str, dst: str) -> bool:
        try:
            ref = str(ref or "").strip()
            if ref.startswith("data:image/"):
                with open(dst, "wb") as fh:
                    fh.write(base64.b64decode(ref.split("base64,", 1)[-1]))
                return True
            if ref.startswith("file://"):
                ref = ref[len("file://"):]
            if ref.startswith(("http://", "https://")):
                import requests
                r = requests.get(ref, timeout=30, proxies={"http": None, "https": None})
                r.raise_for_status()
                with open(dst, "wb") as fh:
                    fh.write(r.content)
                return True
            if os.path.exists(ref):
                shutil.copyfile(ref, dst)
                return True
        except Exception as exc:
            print(f"[mm] failed to materialize image ref {ref[:80]!r}: {exc}")
        return False

    def _ensure_mm_registry(self):
        tl = self._get_thread_state()
        refs = list(getattr(tl, "image_refs", None) or [])
        key = (tuple(refs), getattr(tl, "task_log_dir", None))
        if tl.mm_registry is not None and tl.mm_key == key:
            return tl.mm_registry, tl.mm_workdir
        if getattr(tl, "task_log_dir", None):
            workdir = os.path.join(tl.task_log_dir, "mm_images")
        else:
            workdir = tempfile.mkdtemp(prefix="mm_images_")
        os.makedirs(workdir, exist_ok=True)
        registry = {}
        assets = {}
        tl.mm_seen_perceptual_hashes = []
        for i, ref in enumerate(refs, start=1):
            dst = os.path.join(workdir, f"img_{i}")
            if self._materialize_image_ref(ref, dst):
                handle = f"img_{i}"
                registry[handle] = dst
                assets[handle] = {
                    "handle": handle,
                    "kind": "question_image",
                    "local_path": dst,
                    "source_ref": (
                        "data:image/..." if str(ref).startswith("data:image/") else ref
                    ),
                }
                fingerprint = self._perceptual_hash(dst)
                if fingerprint is not None:
                    tl.mm_seen_perceptual_hashes.append(fingerprint)
        tl.mm_registry, tl.mm_workdir, tl.mm_key = registry, workdir, key
        tl.mm_assets = assets
        tl.mm_search_count = 0
        tl.mm_page_count = 0
        tl.mm_failed_handles = {}
        self._persist_mm_registry_manifest()
        return registry, workdir

    def _persist_mm_registry_manifest(self):
        tl = self._get_thread_state()
        registry = getattr(tl, "mm_registry", None) or {}
        workdir = getattr(tl, "mm_workdir", None)
        if not workdir:
            return
        assets = getattr(tl, "mm_assets", None)
        if assets is None:
            assets = {}
            tl.mm_assets = assets
        for handle, path in registry.items():
            assets.setdefault(handle, {
                "handle": handle,
                "kind": self._handle_kind(handle),
                "local_path": path,
            })
        payload = {
            "version": 1,
            "assets": {key: assets[key] for key in sorted(assets)},
        }
        temp_path = os.path.join(workdir, ".registry.json.tmp")
        final_path = os.path.join(workdir, "registry.json")
        try:
            with open(temp_path, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, default=str)
            os.replace(temp_path, final_path)
        except Exception as exc:
            print(f"[mm] failed to persist image registry: {exc}")


    @staticmethod
    def _mm_image_part(path: str) -> Optional[Dict]:
        if not path or not os.path.isfile(path):
            return None
        return {
            "type": "image_url",
            "image_url": {"url": f"file://{os.path.abspath(path)}"},
        }

    @staticmethod
    def _handle_kind(handle: str) -> str:
        for prefix, kind in (("py_img_", "derived_image"),
                             ("page_img_", "page_image"),
                             ("search_img_", "search_image"),
                             ("img_", "question_image")):
            if str(handle).startswith(prefix):
                return kind
        return "registered_image"

    @staticmethod
    def _mm_image_label(path: str) -> str:
        return os.path.splitext(os.path.basename(path))[0]

    @staticmethod
    def _perceptual_hash(path: str):
        try:
            from PIL import Image

            with Image.open(path) as image:
                resized = image.convert("L").resize((9, 8))
                get_pixels = getattr(resized, "get_flattened_data", resized.getdata)
                pixels = list(get_pixels())
            value = 0
            for row in range(8):
                offset = row * 9
                for column in range(8):
                    value = (value << 1) | int(
                        pixels[offset + column] > pixels[offset + column + 1]
                    )
            return value
        except Exception:
            return None

    @staticmethod
    def _near_duplicate(fingerprint, seen, threshold: int = 5) -> bool:
        return fingerprint is not None and any(
            (fingerprint ^ previous).bit_count() <= threshold for previous in seen
        )

    @staticmethod
    def _attachment_value(attachment, name, default=None):
        if isinstance(attachment, dict):
            return attachment.get(name, default)
        return getattr(attachment, name, default)

    def _register_mm_attachment_metadata(self, result):
        if not self._web_mm_active():
            return
        tl = self._get_thread_state()
        registry, _ = self._ensure_mm_registry()
        max_page_index = int(getattr(tl, "mm_page_count", 0) or 0)
        changed = False
        for attachment in list(getattr(result, "attachments", None) or []):
            handle = (
                self._attachment_value(attachment, "handle")
                or self._attachment_value(attachment, "label")
            )
            if not handle:
                continue
            handle = str(handle)
            record = dict(attachment) if isinstance(attachment, dict) else {}
            if attachment is not None and not record:
                for name in (
                    "page_url", "source_url", "original_name", "caption",
                    "content_hash", "mime_type", "width", "height",
                    "source_width", "source_height", "source_bytes",
                    "encoded_bytes", "was_resized", "cache_hit",
                    "rendition_policy", "kind", "local_path", "path",
                ):
                    value = getattr(attachment, name, None)
                    if value is not None:
                        record[name] = value
            path = record.get("local_path") or record.get("path")
            if path:
                record["local_path"] = str(path)
                registry[handle.casefold()] = str(path)
            record["handle"] = handle
            previous = tl.mm_assets.get(handle) or tl.mm_assets.get(handle.casefold())
            if isinstance(previous, dict):
                for key, value in previous.items():
                    if record.get(key) in (None, "") and value not in (None, ""):
                        record[key] = value
            tl.mm_assets[handle] = record
            tl.mm_assets[handle.casefold()] = record
            changed = True
            match = re.fullmatch(r"page_img_(\d+)", handle, re.IGNORECASE)
            if match:
                max_page_index = max(max_page_index, int(match.group(1)))
        if max_page_index != int(getattr(tl, "mm_page_count", 0) or 0):
            tl.mm_page_count = max_page_index
        if changed:
            self._persist_mm_registry_manifest()

    def _result_image_entries(self, result):
        entries, seen = [], set()
        for attachment in list(getattr(result, "attachments", None) or []):
            path = (
                self._attachment_value(attachment, "local_path")
                or self._attachment_value(attachment, "path")
            )
            if not path or str(path) in seen:
                continue
            handle = (
                self._attachment_value(attachment, "handle")
                or self._attachment_value(attachment, "label")
                or self._mm_image_label(str(path))
            )
            entries.append((str(handle), str(path)))
            seen.add(str(path))
        for path in list(getattr(result, "new_image_paths", None) or []):
            if not path or str(path) in seen:
                continue
            entries.append((self._mm_image_label(str(path)), str(path)))
            seen.add(str(path))
        return entries

    def _attach_mm_images(self, messages, result):
        if not self._web_mm_active() or not messages:
            return messages
        entries = self._result_image_entries(result)
        if not entries:
            return messages
        kept, image_parts = [], []
        registry, _ = self._ensure_mm_registry()
        tl = self._get_thread_state()
        attachments_by_path = {}
        for attachment in list(getattr(result, "attachments", None) or []):
            attachment_path = (
                self._attachment_value(attachment, "local_path")
                or self._attachment_value(attachment, "path")
            )
            if attachment_path:
                attachments_by_path[str(attachment_path)] = attachment
        for handle, pth in entries:
            attachment = attachments_by_path.get(pth)
            fingerprint = self._perceptual_hash(pth)
            part = self._mm_image_part(pth)
            if part:
                kept.append((handle, pth))
                image_parts.append(part)
                registry[handle.casefold()] = pth
                record = dict(attachment) if isinstance(attachment, dict) else {}
                if attachment is not None and not record:
                    for name in (
                        "page_url", "source_url", "original_name", "caption",
                        "content_hash", "mime_type", "width", "height",
                        "source_width", "source_height", "source_bytes",
                        "encoded_bytes", "was_resized", "cache_hit",
                        "rendition_policy", "kind",
                    ):
                        value = getattr(attachment, name, None)
                        if value is not None:
                            record[name] = value
                record.update({"handle": handle, "local_path": pth})
                tl.mm_assets[handle] = record
                if fingerprint is not None:
                    tl.mm_seen_perceptual_hashes.append(fingerprint)
        if not image_parts:
            return messages
        self._persist_mm_registry_manifest()
        last = messages[-1]
        text = last.get("content", "")
        if isinstance(text, str):
            prefix = "".join(f"[{handle}]<image>" for handle, _ in kept)
            last = dict(last, content=[{"type": "text", "text": prefix + text}] + image_parts)
            messages = messages[:-1] + [last]
        return messages

    def _resolve_mm_image_arg(self, tool_args: dict, tool_label: str):
        import re
        image_ref = str(tool_args.get("image") or tool_args.get("image_id") or "").strip()
        if not image_ref:
            return tool_args, {}, None
        registry, _ = self._ensure_mm_registry()
        resolved = registry.get(image_ref.casefold())
        if resolved:
            tool_args = dict(tool_args, image=resolved, image_label=image_ref)
        elif re.fullmatch(
            r"(?:img|search_img|page_img|visit_img|open_img|crop_img)_\d+(?:__[^/\\]+)?",
            image_ref,
            re.IGNORECASE,
        ):
            if not resolved:
                return tool_args, {}, (
                    f"[Tool Error] {tool_label}: unknown image reference "
                    f"{image_ref!r}. Available: {sorted(registry) or 'none'}."
                )
        pixel_feedback = getattr(self, "reverse_image_pixel_feedback_enabled", None)
        if pixel_feedback is None:
            pixel_feedback = self._web_mm_active()
        if not bool(pixel_feedback):
            return tool_args, {}, None
        workdir = self._get_thread_state().mm_workdir
        tl = self._get_thread_state()
        return tool_args, {
            "mm_search_workdir": workdir,
            "mm_search_start": tl.mm_search_count + 1,
        }, None


__all__ = ["MMToolMixin"]
