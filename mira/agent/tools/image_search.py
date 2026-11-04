from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import random
import threading
from dataclasses import dataclass
import time
from typing import List, Optional

import requests
from PIL import Image

from .common import tool_cache
from .common.runtime import ConfigSlot


logger = logging.getLogger(__name__)

__all__ = ["configure", "search_image_records"]


@dataclass(frozen=True)
class _Settings:
    lens_url: str
    lens_upload_endpoint: str
    lens_timeout: int
    upload_timeout: int
    upload_concurrency: int
    max_results: int
    serper_keys: tuple
    request_proxies: Optional[dict]


def _build(cfg) -> _Settings:
    raw = (cfg.search.serper_key_ids or "").strip()
    proxy = (cfg.proxy.visit_fetch_proxy or cfg.proxy.https_proxy or cfg.proxy.http_proxy or "").strip()
    return _Settings(
        lens_url=cfg.image.lens_url,
        lens_upload_endpoint=cfg.image.lens_upload_endpoint,
        lens_timeout=int(cfg.image.lens_timeout),
        upload_timeout=int(cfg.image.upload_timeout),
        upload_concurrency=max(1, int(cfg.image.upload_concurrency or 8)),
        max_results=int(cfg.image.max_results),
        serper_keys=tuple(k.strip() for k in raw.split(",") if k.strip()),
        request_proxies={"http": proxy, "https": proxy} if proxy else None,
    )


def _reset(settings: _Settings) -> None:
    global _UPLOAD_LOCK
    _UPLOAD_LOCK = threading.BoundedSemaphore(settings.upload_concurrency)


_UPLOAD_LOCK = threading.BoundedSemaphore(8)

_SLOT = ConfigSlot("image_search", _build, _reset)
configure = _SLOT.configure
S = _SLOT.get


_B64_PREFIXES = ("iVBORw0KGgo", "/9j/4AAQ", "data:image")
_B64_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")


def _looks_like_base64(v) -> bool:
    if not isinstance(v, str) or len(v) <= 500:
        return False
    if any(v.startswith(p) for p in _B64_PREFIXES):
        return True
    sample = v[:100]
    if all(ch in _B64_ALPHABET for ch in sample):
        try:
            base64.b64decode(sample)
            return True
        except Exception:
            return False
    return False


def _materialize_to_bytes(image_ref: str) -> Optional[bytes]:
    if _looks_like_base64(image_ref):
        try:
            return base64.b64decode(image_ref.split("base64,", 1)[-1])
        except Exception as e:
            logger.warning("[image_search] base64 decode failed: %s", e)
            return None
    if isinstance(image_ref, str) and len(image_ref) < 1024 and os.path.exists(image_ref):
        try:
            with open(image_ref, "rb") as fh:
                return fh.read()
        except Exception as e:
            logger.warning("[image_search] read local file failed: %s", e)
            return None
    return None


def _upload_request(endpoint: str, filename: str, payload: bytes, proxies) -> Optional[str]:
    def post(**kwargs):
        if proxies is not None:
            return requests.post(endpoint, proxies=proxies, **kwargs)
        with requests.Session() as session:
            session.trust_env = False
            return session.post(endpoint, **kwargs)

    if "uguu" in endpoint:
        resp = post(files={"files[]": (filename, payload, "image/png")},
                    timeout=S().upload_timeout)
        if resp.status_code == 200:
            url = ((resp.json().get("files") or [{}])[0].get("url") or "").strip()
            if url.startswith("http"):
                return url
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
    resp = post(data={"reqtype": "fileupload"},
                files={"fileToUpload": (filename, payload, "image/png")},
                timeout=S().upload_timeout)
    if resp.status_code == 200 and resp.text.startswith("http"):
        return resp.text.strip()
    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")


def _upload_to_public_host(image_bytes: bytes, filename: str = "img.png") -> Optional[str]:
    if not S().lens_upload_endpoint:
        return None
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        buf = io.BytesIO()
        (pil.convert("RGB") if pil.mode not in ("RGB", "RGBA") else pil).save(buf, format="PNG")
        payload = buf.getvalue()
    except Exception:
        payload = image_bytes
    last = ""
    routes = [None] + ([S().request_proxies] if S().request_proxies else [])
    RATE_BACKOFF = [15, 30, 60]
    NORMAL_BACKOFF = [1.5, 3.0]
    with _UPLOAD_LOCK:
        for attempt in range(3):
            rate_limited = False
            for proxies in routes:
                try:
                    url = _upload_request(S().lens_upload_endpoint, filename, payload, proxies)
                    if url:
                        return url
                except Exception as exc:
                    last = f"{'proxy' if proxies else 'direct'}: {exc}"
                    s = str(exc)
                    if "429" in s or ("500" in s and "wait" in s.lower()) or "rate limit" in s.lower():
                        rate_limited = True
            if attempt < 2:
                delay = (RATE_BACKOFF if rate_limited else NORMAL_BACKOFF)[attempt]
                time.sleep(delay)
    logger.warning("[image_search] upload failed after retries: %s", last)
    return None


def _search_serper(public_url: str, limit: Optional[int] = None) -> List[dict]:
    if not S().serper_keys:
        raise RuntimeError("SERPER_KEY_IDS / SERPER_KEY_ID not configured for /lens.")
    if not S().lens_url:
        raise RuntimeError("image.lens_url not configured.")
    headers = {"X-API-KEY": random.choice(S().serper_keys), "Content-Type": "application/json"}
    data = _post_with_retry(S().lens_url, headers, {"url": public_url})
    out = []
    raw_limit = S().max_results if limit is None else max(0, int(limit))
    for item in (data.get("organic") or [])[:raw_limit]:
        link = item.get("link", "")
        image_url = item.get("imageUrl", "")
        if not (link and image_url):
            continue
        out.append({
            "title": item.get("title", "Untitled"),
            "image_url": image_url,
            "link": link,
            "source": item.get("source", ""),
            "thumbnail_url": item.get("thumbnailUrl", ""),
        })
    return out




def _post_with_retry(url: str, headers: dict, body: dict) -> dict:
    last = None
    for attempt in range(3):
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=S().lens_timeout,
                proxies=S().request_proxies,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(min(1.5 ** attempt + random.uniform(0, 1.0), 15.0))
                continue
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last = e
            if attempt < 2:
                time.sleep(min(1.5 ** attempt + random.uniform(0, 1.0), 10.0))
    raise RuntimeError(f"image search request failed after retries: {last}")


def search_image_records(image_ref: str, limit: Optional[int] = None):
    S()
    if not isinstance(image_ref, str) or not image_ref.strip():
        raise ValueError("image reference must be a non-empty URL, path, or base64 string")
    image_ref = image_ref.strip()

    remote = image_ref.startswith(("http://", "https://"))
    image_bytes = None
    if remote:
        identity = image_ref
    else:
        image_bytes = _materialize_to_bytes(image_ref)
        if image_bytes is None:
            raise ValueError("cannot read image reference as a local path or base64 payload")
        identity = "sha256:" + hashlib.sha256(image_bytes).hexdigest()

    effective_limit = S().max_results if limit is None else max(0, int(limit))
    cache_key = tool_cache.key_for(tool_cache.NS_IMAGE_LENS, {
        "identity": identity,
        "backend": "serper",
        "endpoint": S().lens_url,
        "limit": effective_limit,
    })
    cached = tool_cache.get_json(tool_cache.NS_IMAGE_LENS, cache_key)
    if (
        isinstance(cached, dict)
        and isinstance(cached.get("records"), list)
        and cached["records"]
    ):
        return str(cached.get("public_url") or image_ref), cached["records"]

    if remote:
        public_url = image_ref
    else:
        filename = os.path.basename(image_ref) if len(image_ref) < 1024 else "img.png"
        public_url = _upload_to_public_host(image_bytes, filename=filename or "img.png")
        if not public_url:
            raise RuntimeError(
                "failed to publish the local image; configure a reachable "
                "image.lens_upload_endpoint or provide a public image URL"
            )

    records = _search_serper(public_url, limit=limit)
    if limit is not None:
        records = records[:max(0, int(limit))]
    if records:
        tool_cache.put_json(
            tool_cache.NS_IMAGE_LENS,
            cache_key,
            {"public_url": public_url, "records": records},
            meta=identity[:200],
        )
    return public_url, records
