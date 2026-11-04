from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

T = TypeVar("T")

__all__ = ["ConfigSlot", "WebImageFeatures", "proxy_kwargs", "web_image_features"]


@dataclass(frozen=True)
class WebImageFeatures:
    image_workspace: bool
    reverse_image_search: bool
    reverse_image_pixels: bool
    page_image_pixels: bool
    page_image_open: bool
    python_image_auto_feedback: bool

    @property
    def any_tool_pixels(self) -> bool:
        return bool(
            self.reverse_image_pixels
            or self.page_image_pixels
            or self.page_image_open
            or self.python_image_auto_feedback
        )


def _optional_flag(cfg, name: str, fallback: bool) -> bool:
    value = getattr(cfg.infra, name, None)
    return bool(fallback if value is None else value)


def web_image_features(cfg) -> WebImageFeatures:
    legacy = bool(cfg.infra.web_multimodal_enabled)
    legacy_page = legacy and bool(cfg.infra.web_page_images_enabled)
    return WebImageFeatures(
        image_workspace=_optional_flag(cfg, "image_workspace_enabled", legacy),
        reverse_image_search=_optional_flag(
            cfg, "reverse_image_search_enabled", legacy
        ),
        reverse_image_pixels=_optional_flag(
            cfg, "reverse_image_pixel_feedback_enabled", legacy
        ),
        page_image_pixels=_optional_flag(
            cfg, "page_image_pixel_feedback_enabled", legacy_page
        ),
        page_image_open=_optional_flag(
            cfg, "page_image_open_enabled", legacy_page
        ),
        python_image_auto_feedback=_optional_flag(
            cfg, "python_image_auto_feedback_enabled", legacy
        ),
    )


class ConfigSlot(Generic[T]):
    __slots__ = ("_name", "_build", "_on_configure", "_value")

    def __init__(
        self,
        name: str,
        build: Callable[[object], T],
        on_configure: Optional[Callable[[T], None]] = None,
    ) -> None:
        self._name = name
        self._build = build
        self._on_configure = on_configure
        self._value: Optional[T] = None

    def configure(self, cfg) -> None:
        value = self._build(cfg)
        self._value = value
        if self._on_configure is not None:
            self._on_configure(value)

    def get(self) -> T:
        if self._value is None:
            raise RuntimeError(
                f"configure_engine(cfg) not called ({self._name} used before configure)"
            )
        return self._value

    @property
    def ready(self) -> bool:
        return self._value is not None


def proxy_kwargs(cfg) -> dict:
    proxy = (
        cfg.proxy.visit_fetch_proxy
        or cfg.proxy.https_proxy
        or cfg.proxy.http_proxy
    )
    return {"proxies": {"http": proxy, "https": proxy}} if proxy else {}
