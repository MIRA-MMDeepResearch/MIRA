import asyncio
import io
import logging
import os
import re
import threading
import sys
import time
from dataclasses import dataclass
from urllib.parse import unquote, urljoin, urlparse

from ...timing import tool_subphase

from .runtime import ConfigSlot, proxy_kwargs

__all__ = ["local_fetch_sync", "configure"]

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class _Settings:
    hard_timeout: int
    per_tier_timeout: int
    pdf_max_pages: int
    playwright_concurrency: int
    proxy_kwargs: dict
    playwright_proxy: dict


def _playwright_proxy(kwargs: dict) -> dict:
    url = (kwargs.get("proxies") or {}).get("https") or (kwargs.get("proxies") or {}).get("http")
    if not url:
        return {}
    parsed = urlparse(url)
    if not parsed.hostname:
        return {}
    port = f":{parsed.port}" if parsed.port else ""
    out = {"server": f"{parsed.scheme or 'http'}://{parsed.hostname}{port}"}
    if parsed.username:
        out["username"] = unquote(parsed.username)
    if parsed.password:
        out["password"] = unquote(parsed.password)
    return out


def _build(cfg) -> _Settings:
    kwargs = proxy_kwargs(cfg)
    return _Settings(
        hard_timeout=int(cfg.fetch.timeout),
        per_tier_timeout=int(cfg.fetch.tier_timeout),
        pdf_max_pages=int(cfg.fetch.pdf_max_pages),
        playwright_concurrency=int(cfg.fetch.playwright_concurrency),
        proxy_kwargs=kwargs,
        playwright_proxy=_playwright_proxy(kwargs),
    )


def _reset(settings: _Settings) -> None:
    global _PLAYWRIGHT_SEM
    _PLAYWRIGHT_SEM = threading.Semaphore(settings.playwright_concurrency)


_PLAYWRIGHT_OK: "bool | None" = None
_CHROMIUM_ARGS = ["--no-sandbox"]
_PLAYWRIGHT_SEM = threading.Semaphore(4)
_HTML_PARSE_LOCK = threading.BoundedSemaphore(8)

_SLOT = ConfigSlot("local_fetch", _build, _reset)
configure = _SLOT.configure
S = _SLOT.get


def _ensure_browser_libs() -> None:
    lib = os.path.join(sys.prefix, "lib")
    if not os.path.isdir(lib):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if lib not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{lib}:{current}" if current else lib


async def _playwright_available() -> bool:
    global _PLAYWRIGHT_OK
    if _PLAYWRIGHT_OK is not None:
        return _PLAYWRIGHT_OK
    try:
        import importlib.util
        if importlib.util.find_spec("playwright") is None:
            raise ImportError("playwright package not installed")
        _ensure_browser_libs()
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=_CHROMIUM_ARGS)
            await browser.close()
        _PLAYWRIGHT_OK = True
    except Exception as e:
        _PLAYWRIGHT_OK = False
        logger.warning(
            "[local_fetch] Tier 3 (playwright JS render) DISABLED: %s. "
            "Chain runs as 3 tiers (curl_cffi + PDF + archive); JS/SPA pages fall to archive. "
            "To enable: pip install playwright && playwright install chromium.",
            e,
        )
    return _PLAYWRIGHT_OK


def _strip_html(body: str, page_url: str = "") -> str:
    with _HTML_PARSE_LOCK:
        try:
            import trafilatura
            out = trafilatura.extract(
                body,
                url=page_url or None,
                output_format="markdown",
                include_comments=False,
                include_tables=True,
                include_images=True,
                include_links=True,
                favor_recall=True,
            )
            if out:
                return out
        except Exception:
            pass
    try:
        from bs4 import BeautifulSoup, NavigableString

        soup = BeautifulSoup(body or "", "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        for img in soup.find_all("img"):
            src = (img.get("src") or "").strip()
            if not src:
                img.decompose()
                continue
            alt = (
                img.get("alt")
                or img.get("title")
                or img.get("aria-label")
                or "Image"
            )
            img.replace_with(NavigableString(f" ![{alt}]({urljoin(page_url, src)}) "))

        for a in soup.find_all("a"):
            href = (a.get("href") or "").strip()
            text = a.get_text(" ", strip=True)
            if href and text:
                a.replace_with(NavigableString(f" [{text}]({urljoin(page_url, href)}) "))
            elif text:
                a.replace_with(NavigableString(f" {text} "))
            else:
                a.decompose()

        return soup.get_text(" ", strip=False)
    except Exception:
        return re.sub(r"<[^>]+>", " ", body or "")


def _parse_pdf(raw: bytes) -> str:
    try:
        import pypdf
        rd = pypdf.PdfReader(io.BytesIO(raw))
        pages = rd.pages[:S().pdf_max_pages]
        return "\n".join((p.extract_text() or "") for p in pages)
    except Exception:
        return ""


def _is_pdf(url: str, content_type: str, head: bytes) -> bool:
    return (
        url.lower().endswith(".pdf")
        or "application/pdf" in (content_type or "").lower()
        or head[:5] == b"%PDF-"
    )


async def _local_fetch_async(url: str, timeout: int) -> str:
    from curl_cffi.requests import AsyncSession

    tier_to = min(S().per_tier_timeout, timeout)
    chain_deadline = time.monotonic() + timeout
    proxies = {}
    proxies = S().proxy_kwargs or proxies

    async def _curl(target, imp, want_bytes=False):
        try:
            async with AsyncSession() as s:
                with tool_subphase(f"local_fetch.http.{imp}"):
                    r = await s.get(target, impersonate=imp, timeout=tier_to,
                                    allow_redirects=True, **proxies)
            return r.status_code, (r.content if want_bytes else r.text), r.headers.get("content-type", "")
        except Exception:
            return 0, (b"" if want_bytes else ""), ""

    st, raw, ct = await _curl(url, "chrome131", want_bytes=True)
    if st in (403, 502, 503, 520) or st == 0:
        st2, raw2, ct2 = await _curl(url, "safari17_0", want_bytes=True)
        if st2 == 200:
            st, raw, ct = st2, raw2, ct2

    if st == 200 and raw:
        if _is_pdf(url, ct, raw[:5]):
            txt = _parse_pdf(raw)
            if len(txt.strip()) >= 200:
                return txt
        else:
            txt = _strip_html(raw.decode("utf-8", "ignore"), url)
            if len(txt.strip()) >= 300:
                return txt

    if chain_deadline - time.monotonic() >= 3.0 and await _playwright_available():
        with tool_subphase("local_fetch.playwright_queue"):
            got_slot = _PLAYWRIGHT_SEM.acquire(timeout=tier_to)
        if got_slot:
            try:
                from playwright.async_api import async_playwright
                with tool_subphase("local_fetch.playwright"):
                    async with async_playwright() as pw:
                        br = await pw.chromium.launch(
                            headless=True, args=_CHROMIUM_ARGS,
                            proxy=S().playwright_proxy or None,
                        )
                        try:
                            pg = await br.new_page()
                            await pg.goto(
                                url, timeout=tier_to * 1000, wait_until="domcontentloaded"
                            )
                            html = await pg.content()
                        finally:
                            await br.close()
                t3 = _strip_html(html, url)
                if len(t3.strip()) >= 200:
                    return t3
            except Exception:
                pass
            finally:
                _PLAYWRIGHT_SEM.release()

    if chain_deadline - time.monotonic() >= 2.0:
        snaps = await asyncio.gather(
            *(_curl(f"https://web.archive.org/web/{yr}/{url}", "chrome131")
              for yr in ("2024", "2023")),
            return_exceptions=True,
        )
        for snap in snaps:
            if isinstance(snap, BaseException):
                continue
            s4, body4, _ = snap
            if s4 == 200:
                t4 = _strip_html(body4, url)
                if len(t4.strip()) >= 200:
                    return t4

    return f"[visit] Error: local fetch failed for {url} (all tiers exhausted)"


def local_fetch_sync(url: str, timeout: int = None) -> str:
    if timeout is None:
        timeout = S().hard_timeout

    async def _run():
        return await asyncio.wait_for(_local_fetch_async(url, timeout), timeout=timeout)

    try:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_run())
        finally:
            loop.close()
    except asyncio.TimeoutError:
        return f"[visit] Error: local fetch timed out after {timeout}s for {url}"
    except Exception as e:
        return f"[visit] Error: local fetch crashed for {url}: {type(e).__name__}: {str(e)[:200]}"
