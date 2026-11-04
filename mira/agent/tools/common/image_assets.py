from __future__ import annotations

import hashlib
import io
import ipaddress
import math
import os
import re
import socket
import threading
from dataclasses import asdict, dataclass
from typing import Mapping, Optional
from urllib.parse import unquote, urljoin, urlparse

from . import tool_cache


_MAX_IMAGE_BYTES = 15 * 1024 * 1024
_MIN_IMAGE_BYTES = 128
_MAX_DECODED_PIXELS = 40 * 1024 * 1024
_DEFAULT_MAX_EDGE = 1024
_DEFAULT_MAX_PIXELS = 1024 * 1024
_DEFAULT_OUTPUT_MAX_BYTES = 1024 * 1024
_DEFAULT_JPEG_QUALITY = 75
_RENDITION_POLICY_VERSION = "jpeg-one-megapixel-v1"
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_IMAGE_DOWNLOAD_SLOTS = threading.BoundedSemaphore(16)
_IMAGE_CACHE_LOCKS = [threading.Lock() for _ in range(64)]


@dataclass(frozen=True)
class PageImage:
    source_url: str
    page_url: str
    marker: str
    original_name: str = ""
    alt: str = ""
    caption: str = ""
    title: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "PageImage":
        fields = cls.__dataclass_fields__
        return cls(**{key: str(value.get(key) or "") for key in fields})


@dataclass(frozen=True)
class ImageAttachment:
    handle: str
    local_path: str
    page_url: str
    source_url: str
    original_name: str
    alt: str = ""
    caption: str = ""
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    content_hash: str = ""
    mime_type: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    source_width: Optional[int] = None
    source_height: Optional[int] = None
    source_bytes: Optional[int] = None
    encoded_bytes: Optional[int] = None
    was_resized: bool = False
    cache_hit: bool = False
    rendition_policy: str = ""


def image_name_from_url(url: str, fallback: str = "image") -> str:
    name = unquote(os.path.basename(urlparse(str(url or "")).path)).strip()
    name = _SAFE_NAME.sub("_", name).strip("._")
    if not name:
        name = fallback
    return name[:120]


def resolve_image_url(page_url: str, value: str) -> str:
    candidate = str(value or "").strip()
    if not candidate or candidate.startswith(("data:", "blob:", "javascript:")):
        return ""
    resolved = urljoin(page_url, candidate)
    return resolved if urlparse(resolved).scheme.lower() in {"http", "https"} else ""


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("image URL must be public HTTP(S)")
    host = parsed.hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        raise ValueError("local image hosts are not allowed")
    try:
        default_port = 443 if parsed.scheme.lower() == "https" else 80
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(host, parsed.port or default_port)
        }
    except OSError as exc:
        raise ValueError(f"cannot resolve image host: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if not ip.is_global:
            raise ValueError("private or non-routable image hosts are not allowed")


def _response_bytes(response, max_bytes: int) -> bytes:
    content_type = str(response.headers.get("content-type") or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise ValueError(f"unexpected image content type: {content_type}")
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            declared_length = int(content_length)
        except (TypeError, ValueError):
            declared_length = 0
        if declared_length > max_bytes:
            raise ValueError("image exceeds byte limit")
    raw = bytearray()
    for chunk in response.iter_content(64 * 1024):
        raw.extend(chunk)
        if len(raw) > max_bytes:
            raise ValueError("image exceeds byte limit")
    return bytes(raw)


def _download_public_image(
    url: str,
    referer: str,
    max_bytes: int,
    proxy_kwargs: Optional[Mapping] = None,
) -> bytes:
    headers = {"User-Agent": "Mozilla/5.0"}
    if referer.startswith(("http://", "https://")):
        headers["Referer"] = referer
    configured_route = dict(proxy_kwargs or {})
    routes = [configured_route, {}] if configured_route else [{}]
    current = url
    for _ in range(4):
        _validate_public_url(current)
        response = None
        statuses = []
        transport_errors = []
        try:
            from curl_cffi import requests as curl_requests
        except ImportError:
            curl_requests = None

        import requests

        session = requests.Session()
        session.trust_env = False
        for route in routes:
            if curl_requests is not None:
                for impersonate in ("chrome131", "safari17_0"):
                    try:
                        response = curl_requests.get(
                            current,
                            headers=headers,
                            timeout=(3, 10),
                            allow_redirects=False,
                            stream=True,
                            impersonate=impersonate,
                            **route,
                        )
                        statuses.append(int(response.status_code))
                        if response.status_code < 400:
                            break
                        response.close()
                        response = None
                    except Exception as exc:
                        transport_errors.append(
                            f"curl_cffi:{type(exc).__name__}"
                        )
                        response = None
            if response is None:
                try:
                    response = session.get(
                        current,
                        headers=headers,
                        timeout=(3, 10),
                        allow_redirects=False,
                        stream=True,
                        **route,
                    )
                    statuses.append(int(response.status_code))
                    if response.status_code < 400:
                        break
                    response.close()
                    response = None
                except Exception as exc:
                    transport_errors.append(f"requests:{type(exc).__name__}")
                    response = None
            if response is not None:
                break
        if response is None:
            if statuses:
                detail = f" (HTTP {statuses[-1]})"
            elif transport_errors:
                detail = " (transport: " + ", ".join(sorted(set(transport_errors))) + ")"
            else:
                detail = ""
            raise ValueError(f"image download failed{detail}")
        try:
            if 300 <= response.status_code < 400:
                target = response.headers.get("location") or ""
                if not target:
                    raise ValueError("image redirect has no location")
                current = urljoin(current, target)
                continue
            if response.status_code >= 400:
                raise ValueError(f"image download failed (HTTP {response.status_code})")
            return _response_bytes(response, max_bytes)
        finally:
            response.close()
    raise ValueError("too many image redirects")


def _looks_like_svg(raw: bytes) -> bool:
    head = raw[:1024].lstrip(b"\xef\xbb\xbf\x00\x09\x0a\x0d\x20").lower()
    return head.startswith(b"<svg") or b"<svg" in head


def _rasterize_svg(raw: bytes) -> bytes:
    try:
        import fitz
    except ImportError as exc:
        raise ValueError("SVG image support is unavailable") from exc
    try:
        document = fitz.open(stream=raw, filetype="svg")
        page = document.load_page(0)
        rect = page.rect
        width, height = int(round(rect.width)), int(round(rect.height))
        if width < 32 or height < 32:
            raise ValueError("image dimensions are too small")
        scale = min(1.0, 4096.0 / max(width, height))
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=True)
        return pixmap.tobytes("png")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid SVG image") from exc


def _model_rendition(
    raw: bytes,
    *,
    max_edge: int = _DEFAULT_MAX_EDGE,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    output_max_bytes: int = _DEFAULT_OUTPUT_MAX_BYTES,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
) -> tuple[bytes, dict]:
    from PIL import Image, ImageOps

    max_edge = max(32, int(max_edge))
    max_pixels = max(32 * 32, int(max_pixels))
    output_max_bytes = max(16 * 1024, int(output_max_bytes))
    jpeg_quality = max(30, min(95, int(jpeg_quality)))
    prepared = _rasterize_svg(raw) if _looks_like_svg(raw) else raw
    try:
        with Image.open(io.BytesIO(prepared)) as opened:
            source_width, source_height = opened.size
            if source_width < 32 or source_height < 32:
                raise ValueError("image dimensions are too small")
            if source_width * source_height > _MAX_DECODED_PIXELS:
                raise ValueError("image decoded pixel count exceeds safety limit")
            try:
                opened.seek(0)
            except EOFError:
                pass
            image = ImageOps.exif_transpose(opened).convert("RGBA")
            image.load()
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("invalid or unsupported image") from exc

    scale = min(
        1.0,
        max_edge / max(source_width, source_height),
        math.sqrt(max_pixels / (source_width * source_height)),
    )
    if scale < 1.0:
        target = (
            max(1, int(source_width * scale)),
            max(1, int(source_height * scale)),
        )
        image = image.resize(target, Image.Resampling.LANCZOS)
    if "A" in image.getbands():
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    else:
        image = image.convert("RGB")

    qualities = []
    for quality in (jpeg_quality, 65, 55, 45, 35, 30):
        if quality not in qualities and quality <= jpeg_quality:
            qualities.append(quality)
    for shrink_round in range(7):
        for quality in qualities:
            stream = io.BytesIO()
            image.save(
                stream,
                format="JPEG",
                quality=quality,
                optimize=True,
                progressive=True,
            )
            encoded = stream.getvalue()
            if len(encoded) <= output_max_bytes:
                return encoded, {
                    "source_width": source_width,
                    "source_height": source_height,
                    "width": image.width,
                    "height": image.height,
                    "source_bytes": len(raw),
                    "encoded_bytes": len(encoded),
                    "was_resized": (
                        image.width != source_width
                        or image.height != source_height
                        or len(encoded) != len(raw)
                    ),
                    "jpeg_quality": quality,
                }
        if shrink_round < 6:
            image = image.resize(
                (
                    max(32, int(image.width * 0.85)),
                    max(32, int(image.height * 0.85)),
                ),
                Image.Resampling.LANCZOS,
            )
    raise ValueError("image cannot fit rendition byte limit")


def _normalise_image(raw: bytes) -> bytes:
    return _model_rendition(raw)[0]


def _inspect_image(raw: bytes) -> tuple[str, str, int, int]:
    from PIL import Image

    if len(raw) < _MIN_IMAGE_BYTES:
        raise ValueError("image is too small")
    with Image.open(io.BytesIO(raw)) as image:
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        width, height = image.size
        fmt = str(image.format or "").upper()
    if width < 32 or height < 32:
        raise ValueError("image dimensions are too small")
    if width * height > _MAX_DECODED_PIXELS:
        raise ValueError("image decoded pixel count exceeds safety limit")
    formats = {
        "JPEG": (".jpg", "image/jpeg"),
        "PNG": (".png", "image/png"),
        "WEBP": (".webp", "image/webp"),
        "GIF": (".gif", "image/gif"),
        "AVIF": (".avif", "image/avif"),
    }
    if fmt not in formats:
        raise ValueError(f"unsupported image format: {fmt or 'unknown'}")
    extension, mime = formats[fmt]
    return extension, mime, width, height


def materialize_image(
    image: PageImage,
    workdir: str,
    handle: str,
    *,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    max_bytes: int = _MAX_IMAGE_BYTES,
    max_edge: int = _DEFAULT_MAX_EDGE,
    max_pixels: int = _DEFAULT_MAX_PIXELS,
    output_max_bytes: int = _DEFAULT_OUTPUT_MAX_BYTES,
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY,
    proxy_kwargs: Optional[Mapping] = None,
) -> ImageAttachment:
    source_url = resolve_image_url(image.page_url, image.source_url)
    if not source_url:
        raise ValueError("page image has no downloadable URL")
    policy = {
        "version": _RENDITION_POLICY_VERSION,
        "max_edge": max(32, int(max_edge)),
        "max_pixels": max(32 * 32, int(max_pixels)),
        "output_max_bytes": max(16 * 1024, int(output_max_bytes)),
        "jpeg_quality": max(30, min(95, int(jpeg_quality))),
    }
    cache_key = tool_cache.key_for(
        tool_cache.NS_IMAGE_BLOB,
        {"url": source_url, "rendition": policy},
    )
    raw = tool_cache.get_blob(tool_cache.NS_IMAGE_BLOB, cache_key)
    cache_hit = raw is not None
    rendition_meta = {}
    if raw is not None and len(raw) > policy["output_max_bytes"]:
        raise ValueError("cached image exceeds rendition byte limit")
    if raw is None:
        lock_index = int(cache_key[:2], 16) % len(_IMAGE_CACHE_LOCKS)
        with _IMAGE_CACHE_LOCKS[lock_index]:
            raw = tool_cache.get_blob(tool_cache.NS_IMAGE_BLOB, cache_key)
            cache_hit = raw is not None
            if raw is None:
                with _IMAGE_DOWNLOAD_SLOTS:
                    source = _download_public_image(
                        source_url,
                        image.page_url,
                        max_bytes,
                        proxy_kwargs=proxy_kwargs,
                    )
                raw, rendition_meta = _model_rendition(
                    source,
                    max_edge=policy["max_edge"],
                    max_pixels=policy["max_pixels"],
                    output_max_bytes=policy["output_max_bytes"],
                    jpeg_quality=policy["jpeg_quality"],
                )
                _inspect_image(raw)
                tool_cache.put_blob(
                    tool_cache.NS_IMAGE_BLOB,
                    cache_key,
                    raw,
                    meta=source_url,
                )
    extension, mime, width, height = _inspect_image(raw)
    content_hash = hashlib.sha256(raw).hexdigest()
    original = image.original_name or image_name_from_url(source_url)
    stem = os.path.splitext(image_name_from_url(original))[0] or "image"
    digest = hashlib.sha256(source_url.encode("utf-8", "ignore")).hexdigest()[:10]
    os.makedirs(workdir, exist_ok=True)
    local_path = os.path.join(workdir, f"{handle}__{stem}__{digest}{extension}")
    if not os.path.exists(local_path) or os.path.getsize(local_path) != len(raw):
        with open(local_path, "wb") as output:
            output.write(raw)
    return ImageAttachment(
        handle=handle,
        local_path=local_path,
        page_url=image.page_url,
        source_url=source_url,
        original_name=original,
        alt=image.alt,
        caption=image.caption,
        line_start=line_start,
        line_end=line_end,
        content_hash=content_hash,
        mime_type=mime,
        width=width,
        height=height,
        source_width=rendition_meta.get("source_width", width),
        source_height=rendition_meta.get("source_height", height),
        source_bytes=rendition_meta.get("source_bytes"),
        encoded_bytes=len(raw),
        was_resized=bool(rendition_meta.get("was_resized", False)),
        cache_hit=cache_hit,
        rendition_policy=_RENDITION_POLICY_VERSION,
    )


__all__ = [
    "ImageAttachment",
    "PageImage",
    "image_name_from_url",
    "materialize_image",
    "resolve_image_url",
]
