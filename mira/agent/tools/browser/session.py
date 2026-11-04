from __future__ import annotations

from ..common.web_errors import is_web_failure

import html
import itertools
import json
import os
import random
import re
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote, unquote, urljoin, urlparse

import requests

from ..common import tool_cache
from ..common.image_assets import ImageAttachment, PageImage, image_name_from_url, materialize_image, resolve_image_url
from ..common.runtime import ConfigSlot, proxy_kwargs
from ..common.serper_keys import SerperKeyPool

_SERPER_SEARCH_BUDGET_S = 90.0


def _html_parse_concurrency() -> int:
    raw = os.environ.get("MIRA_HTML_PARSE_CONCURRENCY", "8")
    try:
        return max(1, min(int(raw), 32))
    except (TypeError, ValueError):
        return 8


_HTML_PARSE_LOCK = threading.BoundedSemaphore(_html_parse_concurrency())
_ENCODING = None
_MODEL_TOKENIZER = None
_MODEL_TOKENIZER_PATH = None


def _load_view_tokenizer(path: str):
    try:
        from tokenizers import Tokenizer

        src = path if path.endswith(".json") else os.path.join(path, "tokenizer.json")
        if os.path.exists(src):
            hf = Tokenizer.from_file(src)
            return lambda text: len(hf.encode(text, add_special_tokens=False).ids)
    except Exception as exc:
        print(f"[browser] view tokenizer {path}: HF load failed ({exc})")

    try:
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        src = path if path.endswith(".model") else os.path.join(path, "tiktoken.model")
        if os.path.exists(src):
            ranks = load_tiktoken_bpe(src)
            enc = tiktoken.Encoding(
                name=f"view:{os.path.basename(os.path.dirname(src)) or 'model'}",
                pat_str=r"""[\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Latin}]+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
                mergeable_ranks=ranks,
                special_tokens={},
            )
            return lambda text: len(enc.encode(text, disallowed_special=()))
    except Exception as exc:
        print(f"[browser] view tokenizer {path}: tiktoken load failed ({exc})")

    print(f"[browser] view tokenizer {path} unusable; using o200k_base")
    return None


def _truncate_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return text
    if _view_token_len(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _view_token_len(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    cut = text[:lo]
    space = cut.rfind(" ")
    return (cut[:space] if space > lo * 0.8 else cut).rstrip() + " …"


def _view_token_len(text: str) -> int:
    global _ENCODING, _MODEL_TOKENIZER, _MODEL_TOKENIZER_PATH
    try:
        path = S().tokenizer_path
    except Exception:
        path = ""
    if path:
        if _MODEL_TOKENIZER is None or _MODEL_TOKENIZER_PATH != path:
            _MODEL_TOKENIZER = _load_view_tokenizer(path)
            _MODEL_TOKENIZER_PATH = path
        if _MODEL_TOKENIZER is not None:
            return _MODEL_TOKENIZER(text)
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("o200k_base")
    return len(_ENCODING.encode(text, disallowed_special=()))


@dataclass(frozen=True)
class _Settings:
    serper_key_pool: SerperKeyPool
    proxy_kwargs: dict
    result_snippet_tokens: int
    view_tokens: int
    jina_api_keys: str
    jina_read_base: str
    fetch_timeout: int
    web_image_max_per_call: int
    web_image_max_per_page: int
    web_image_max_bytes: int
    tokenizer_path: str


def _build(cfg) -> _Settings:
    keys = [k.strip() for k in (cfg.search.serper_key_ids or "").split(",") if k.strip()]
    return _Settings(
        serper_key_pool=SerperKeyPool(keys),
        proxy_kwargs=proxy_kwargs(cfg),
        result_snippet_tokens=max(0, int(cfg.search.result_snippet_tokens or 0)),
        view_tokens=max(0, int(cfg.search.view_tokens or 0)),
        jina_api_keys=cfg.search.jina_api_keys or "",
        jina_read_base=(cfg.fetch.jina_read_base or "https://r.jina.ai").rstrip("/"),
        fetch_timeout=int(cfg.fetch.timeout or 60),
        web_image_max_per_call=max(0, int(cfg.infra.web_image_max_per_call or 0)),
        web_image_max_per_page=max(0, int(cfg.infra.web_image_max_per_page or 0)),
        web_image_max_bytes=max(1, int(cfg.infra.web_image_max_bytes or 1)),
        tokenizer_path=str(cfg.context.tokenizer_path or ""),
    )


_SLOT = ConfigSlot("browser_session", _build)
configure = _SLOT.configure
S = _SLOT.get


def _next_serper_key() -> str:
    pool = S().serper_key_pool
    if not pool:
        raise BrowserUsageError("SERPER_API_KEY is not configured.")
    return pool.acquire()


def web_image_limits() -> Tuple[int, int, int]:
    per_call = S().web_image_max_per_call
    per_page = S().web_image_max_per_page
    max_bytes = S().web_image_max_bytes
    return per_call, per_page, max_bytes


class BrowserUsageError(Exception):
    pass


@dataclass(frozen=True)
class Extract:
    url: str
    text: str
    title: str
    line_idx: Optional[int] = None


@dataclass
class BrowserPage:
    url: str
    title: str
    text: str
    urls: Dict[str, str] = field(default_factory=dict)
    snippets: Optional[Dict[str, Extract]] = None
    images: List[PageImage] = field(default_factory=list)


class BrowserResult(str):
    def __new__(cls, text: str, attachments: Sequence[ImageAttachment] = ()):
        obj = super().__new__(cls, text)
        obj.attachments = list(attachments)
        paths = []
        for attachment in obj.attachments:
            if isinstance(attachment, dict):
                path = attachment.get("local_path") or attachment.get("path")
            else:
                path = getattr(attachment, "local_path", "")
            if path:
                paths.append(str(path))
        obj.new_image_paths = paths
        return obj


def _wrap_lines(text: str, width: int = 80) -> List[str]:
    return list(
        itertools.chain.from_iterable(
            textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            if line
            else [""]
            for line in text.split("\n")
        )
    )


_PARTIAL_INITIAL_LINK = re.compile(r"^[^【】]*】")
_PARTIAL_FINAL_LINK = re.compile(r"【\d*(?:†(?P<content>[^†】]*)(?:†[^†】]*)?)?$")
_LINK = re.compile(r"【\d+†(?P<content>[^†】]+)(?:†[^†】]+)?】")


_ANALYTICS_URL = re.compile(
    r"/tr\?|[?&]ev=PageView|[?&]noscript=1"
    r"|/b/ss/|smetrics\.|omtrdc\.net"
    r"|google-analytics\.com|/__utm\.|/collect\?|doubleclick\.net"
    r"|/akam/\d+/pixel_|//pixel\.|/pixel\.gif|/p\.gif\?"
    r"|visitor-counter|/beacon\b|/telemetry\b|/impression\b|scorecardresearch",
    re.I,
)

_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_BARE_URL = re.compile(r"https?://\S+|\S*\.(?:com|org|net|io|co\.uk)/\S*", re.I)
_MD_NOISE = re.compile(r"[#*_`>|]+")
_NAV_WORDS = re.compile(
    r"^(toggle navigation|menu|search|home|share|advertisement|subscribe|sign in|"
    r"log in|follow us|read more|next|previous|skip to content)$", re.I)
_IMAGE_MARKER_LINE = re.compile(
    r"^\s*(?:L\d+:\s*)?(?:-\s*)?(?:\[(?:page_img_\d+|images?[ _]?\d*)[:\]]|images? on page)", re.I)


def _prose_only(line: str) -> str:
    raw = str(line or "")
    if _IMAGE_MARKER_LINE.match(raw):
        return ""
    text = _strip_links(raw)
    text = _MD_LINK.sub(lambda m: m.group(1), text)
    text = _BARE_URL.sub(" ", text)
    text = _MD_NOISE.sub(" ", text)
    text = " ".join(
        tok for tok in text.split()
        if "/" not in tok and tok.count("-") < 3
    )
    text = re.sub(r"\s+", " ", text).strip(" -–—·|()")
    if not text or _NAV_WORDS.match(text):
        return ""
    return text if re.search(r"[a-z]{3}", text) else ""


def _strip_links(text: str) -> str:
    text = _PARTIAL_INITIAL_LINK.sub("", text)
    text = _PARTIAL_FINAL_LINK.sub(lambda match: match.group("content") or "", text)
    return _LINK.sub(lambda match: match.group("content"), text)


class BrowserSession:
    def __init__(
        self,
        search_fn: Optional[Callable[[str, int], List[dict]]] = None,
        fetch_fn: Optional[Callable[[str, bool], BrowserPage]] = None,
        view_tokens: Optional[int] = None,
    ):
        self.pages: List[BrowserPage] = []
        self.pages_by_url: Dict[str, BrowserPage] = {}
        self.search_fn = search_fn or self._cached_search
        self.fetch_fn = fetch_fn or self._cached_fetch
        if view_tokens is None:
            try:
                view_tokens = S().view_tokens
            except Exception:
                view_tokens = 1024
        self.view_tokens = view_tokens

    @property
    def current_cursor(self) -> int:
        return len(self.pages) - 1

    def _get_page(self, cursor: int = -1) -> BrowserPage:
        if not self.pages:
            raise BrowserUsageError("No pages to access!")
        if cursor == -1:
            return self.pages[-1]
        if not isinstance(cursor, int):
            raise BrowserUsageError(
                f"`cursor` should be an integer, not `{type(cursor).__name__}`"
            )
        if cursor < 0 or cursor >= len(self.pages):
            return self.pages[-1]
        return self.pages[cursor]

    def _add_page(self, page: BrowserPage) -> None:
        self.pages.append(page)
        self.pages_by_url[page.url] = page

    @staticmethod
    def _safe_text(value: object) -> str:
        return html.unescape(str(value or "")).replace("【", "〖").replace("】", "〗")

    @staticmethod
    def _domain(url: str) -> str:
        return urlparse(url).netloc

    def _serper_search(self, query: str, topn: int) -> List[dict]:
        chinese = any("\u4e00" <= char <= "\u9fff" for char in query)
        payload = {
            "q": query,
            "num": topn,
            "gl": "cn" if chinese else "us",
            "hl": "zh-cn" if chinese else "en",
        }
        data = None
        pool = S().serper_key_pool
        deadline = time.monotonic() + _SERPER_SEARCH_BUDGET_S
        for attempt in range(8):
            key = _next_serper_key()
            try:
                response = requests.post(
                    "https://google.serper.dev/search",
                    json=payload,
                    headers={
                        "X-API-KEY": key,
                        "Content-Type": "application/json",
                    },
                    timeout=30,
                )
                if response.status_code == 429:
                    try:
                        retry_after = float(response.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        retry_after = 0.0
                    pool.rate_limited(key, retry_after)
                    if time.monotonic() >= deadline:
                        break
                    if not pool.any_available():
                        time.sleep(min(2.0 + attempt, 10.0, max(0.1, deadline - time.monotonic())))
                    continue
                if response.status_code >= 500:
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(min(1.5**attempt + random.random(), 20.0, max(0.1, deadline - time.monotonic())))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except BrowserUsageError:
                raise
            except Exception as exc:
                if attempt == 7 or time.monotonic() >= deadline:
                    raise BrowserUsageError(f"Error during search for `{query}`: {exc}") from exc
                time.sleep(min(1.5**attempt + random.random(), 20.0, max(0.1, deadline - time.monotonic())))
        if data is None:
            raise BrowserUsageError(
                f"Search temporarily rate-limited for `{query}`; retry the same query later."
            )
        results = data.get("organic", []) if isinstance(data, dict) else []
        return results[:topn]

    def _configured_search(self, query: str, topn: int) -> List[dict]:
        return self._serper_search(query, topn)

    def _result_page(self, records: List[dict], title: str, scheme: str) -> str:
        stamp = int(time.time())
        pseudo_url = f"{scheme}://ts={stamp}"
        if pseudo_url in self.pages_by_url:
            pseudo_url = f"{pseudo_url}&n={len(self.pages)}"
        urls: Dict[str, str] = {}
        items = []
        for result in records:
            url = str(result.get("link") or result.get("url") or "").strip()
            if not url:
                continue
            title_text = self._safe_text(result.get("title") or url)
            snippet = self._safe_text(
                result.get("snippet") or result.get("summary") or result.get("description")
            )
            domain = self._domain(url)
            date = self._safe_text(result.get("date"))
            if date:
                snippet = f"{snippet} ({date})" if snippet else date
            marker = f"【{len(urls)}†{title_text}†{domain}】" if domain else f"【{len(urls)}†{title_text}】"
            urls[str(len(urls))] = url
            items.append(f"  * {marker} {snippet}".rstrip())
            content = str(result.get("content") or "")
            if content:
                self.pages_by_url[url] = BrowserPage(
                    url=url,
                    title=title_text,
                    text=f"\nURL: {url}\n{content}",
                    urls={},
                    snippets=None,
                )
        if not items:
            raise BrowserUsageError(f"No results returned for query: `{title}`")

        page = BrowserPage(
            url=pseudo_url,
            title=title,
            text=f"\nURL: {pseudo_url}\n# Search Results\n\n" + "\n".join(items),
            urls=urls,
            snippets={},
        )
        self._add_page(page)
        return self._render(page, self.current_cursor, num_lines=len(_wrap_lines(page.text)))

    def search(self, query: Union[str, List[str]], topn: int = 10) -> str:
        if isinstance(query, list):
            if not query:
                raise BrowserUsageError("Search query cannot be empty.")
            if len(query) == 1:
                return self.search(query[0], topn=topn)
            for item in query:
                if not isinstance(item, str) or not item.strip():
                    raise BrowserUsageError("Search query cannot be empty.")
            clamped = max(1, min(int(topn), 10))
            with ThreadPoolExecutor(max_workers=min(len(query), 5)) as pool:
                futures = [pool.submit(self.search_fn, item, clamped) for item in query]
            blocks = []
            for item, future in zip(query, futures):
                results = future.result()
                if not results:
                    raise BrowserUsageError(f"No results returned for query: `{item}`")
                blocks.append(self._result_page(results, title=item, scheme="web-search"))
            return "\n\n".join(blocks)
        if not isinstance(query, str) or not query.strip():
            raise BrowserUsageError("Search query cannot be empty.")
        topn = max(1, min(int(topn), 10))
        results = self.search_fn(query, topn)
        if not results:
            raise BrowserUsageError(f"No results returned for query: `{query}`")
        return self._result_page(results, title=query, scheme="web-search")

    def search_image(self, records: List[dict], label: str) -> str:
        if not records:
            raise BrowserUsageError(f"No visual matches for image query: `{label}`")
        norm = []
        for r in records:
            norm.append({
                "link": r.get("link") or r.get("url") or "",
                "title": r.get("title") or "",
                "snippet": r.get("source") or "",
            })
        return self._result_page(norm, title=label, scheme="image-search")

    def _process_markdown(self, text: str, url: str, title: str) -> BrowserPage:
        images: List[PageImage] = []
        marker_count = 0
        image_pattern = re.compile(
            r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^\s)]+)(?:\s+[\"'][^\"']*[\"'])?\)"
        )

        def replace(match) -> str:
            nonlocal marker_count
            index = marker_count
            marker_count += 1
            alt = re.sub(r"\s+", " ", match.group("alt") or "").strip()
            source = resolve_image_url(url, match.group("src") or "")
            marker = f"[Image {index}: {alt}]" if alt else f"[Image {index}]"
            if source:
                images.append(
                    PageImage(
                        source_url=source,
                        page_url=url,
                        marker=f"[Image {index}",
                        original_name=image_name_from_url(source),
                        alt=alt,
                    )
                )
            return marker

        processed = image_pattern.sub(replace, text or "")
        return BrowserPage(
            url=url,
            title=title,
            text=f"\nURL: {url}\n{processed}",
            images=images,
        )

    def _fetch_page(self, url: str, view_source: bool = False) -> BrowserPage:
        from curl_cffi import requests as curl_requests

        timeout = S().fetch_timeout
        response = None
        deadline = time.monotonic() + timeout
        for impersonate in ("chrome131", "safari17_0"):
            left = deadline - time.monotonic()
            if impersonate != "chrome131":
                if left < 2 or (response is not None and response.status_code in (401, 404, 410)):
                    break
            try:
                response = curl_requests.get(
                    url,
                    impersonate=impersonate,
                    timeout=max(2, int(left)),
                    allow_redirects=True,
                    **S().proxy_kwargs,
                )
                if response.status_code == 200 and not is_web_failure(
                    str(response.text or "")[:20000], response.status_code, scope="target"
                ):
                    break
            except Exception:
                response = None
        if (
            response is not None
            and response.status_code == 200
            and not is_web_failure(
                str(response.text or "")[:20000], response.status_code, scope="target"
            )
        ):
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" not in content_type and not response.content.startswith(b"%PDF-"):
                if view_source:
                    return BrowserPage(url=url, title=f"view-source:{url}", text=response.text)
                return self._process_html(response.text, str(response.url or url))

        jina_key = S().jina_api_keys.split(",")[0].strip()
        jina_base = S().jina_read_base
        if jina_key:
            try:
                resp_j = requests.get(
                    f"{jina_base}/{url}",
                    headers={"Authorization": f"Bearer {jina_key}"},
                    timeout=timeout,
                    **S().proxy_kwargs,
                )
                if (
                    resp_j.status_code == 200
                    and resp_j.text.strip()
                    and not is_web_failure(
                        resp_j.text[:20000], resp_j.status_code, scope="target"
                    )
                ):
                    text = resp_j.text
                    title = self._domain(url) or url
                    for line in text.splitlines()[:5]:
                        if line.startswith("Title:"):
                            title = line.split(":", 1)[1].strip() or title
                            break
                    return self._process_markdown(text, url, title)
            except Exception:
                pass

        try:
            scrape_key = _next_serper_key()
            resp2 = requests.post(
                "https://scrape.serper.dev",
                json={"url": url},
                headers={"X-API-KEY": scrape_key, "Content-Type": "application/json"},
                timeout=timeout,
            )
            if resp2.status_code == 429:
                S().serper_key_pool.rate_limited(scrape_key)
            if resp2.status_code == 200:
                data = resp2.json()
                text = data.get("text") or data.get("markdown") or ""
                if text.strip() and not is_web_failure(
                    text[:20000], resp2.status_code, scope="target"
                ):
                    meta = data.get("metadata") or {}
                    title = (meta.get("title") if isinstance(meta, dict) else "") or self._domain(url) or url
                    return self._process_markdown(text, url, title)
        except BrowserUsageError:
            raise
        except Exception:
            pass

        from ..common.local_fetch import local_fetch_sync

        text = local_fetch_sync(url, timeout=timeout)
        if text.startswith("[visit] Error"):
            raise BrowserUsageError(f"Error fetching URL `{url}`: {text}")
        title = self._domain(url) or url
        return BrowserPage(url=url, title=title, text=f"\nURL: {url}\n{text}")

    def _search_cache_key(self, query: str, topn: int) -> str:
        return tool_cache.key_for(tool_cache.NS_BROWSER_SEARCH, {
            "query": query,
            "topn": int(topn),
            "backend": "serper",
            "endpoint": "",
            "topk": 0,
        })

    def _cached_search(self, query: str, topn: int) -> List[dict]:
        if not tool_cache.enabled():
            return self._configured_search(query, topn)
        key = self._search_cache_key(query, topn)
        cached = tool_cache.get_json(tool_cache.NS_BROWSER_SEARCH, key)
        if cached:
            return cached
        results = self._configured_search(query, topn)
        if results:
            tool_cache.put_json(tool_cache.NS_BROWSER_SEARCH, key, results, meta=str(query)[:200])
        return results

    def _fetch_cache_key(self, url: str, view_source: bool) -> str:
        jina_key = S().jina_api_keys.split(",")[0].strip()
        return tool_cache.key_for(tool_cache.NS_BROWSER_FETCH, {
            "url": url,
            "view_source": bool(view_source),
            "jina": S().jina_read_base if jina_key else "",
            "page_parser": "images-v3-html-base",
        })

    def _cached_fetch(self, url: str, view_source: bool = False) -> BrowserPage:
        if not tool_cache.enabled():
            return self._fetch_page(url, view_source=view_source)
        key = self._fetch_cache_key(url, view_source)
        cached = tool_cache.get_json(tool_cache.NS_BROWSER_FETCH, key)
        if (
            isinstance(cached, dict)
            and cached.get("text")
            and tool_cache.is_cacheable_text(str(cached["text"]), scope="target")
        ):
            return BrowserPage(
                url=str(cached.get("url") or url),
                title=str(cached.get("title") or ""),
                text=str(cached["text"]),
                urls=dict(cached.get("urls") or {}),
                snippets=None,
                images=[
                    PageImage.from_dict(value)
                    for value in (cached.get("images") or [])
                    if isinstance(value, dict)
                ],
            )
        page = self._fetch_page(url, view_source=view_source)
        if tool_cache.is_cacheable_text(page.text, scope="target"):
            tool_cache.put_json(
                tool_cache.NS_BROWSER_FETCH,
                key,
                {
                    "url": page.url,
                    "title": page.title,
                    "text": page.text,
                    "urls": page.urls,
                    "images": [image.to_dict() for image in page.images],
                },
                meta=url[:400],
            )
        return page

    @staticmethod
    def _replace_node(node, replacement: str) -> None:
        parent = node.getparent()
        if parent is None:
            return
        previous = node.getprevious()
        tail = node.tail or ""
        if previous is None:
            parent.text = (parent.text or "") + replacement + tail
        else:
            previous.tail = (previous.tail or "") + replacement + tail
        parent.remove(node)

    @staticmethod
    def _restore_missing_image_markers(
        text: str, images: Sequence[PageImage]
    ) -> str:
        text = str(text or "")
        missing = [image for image in images if image.marker and image.marker not in text]
        if not missing:
            return text

        replacements: List[Tuple[int, int, str]] = []
        claimed: List[Tuple[int, int]] = []
        unanchored: List[str] = []
        for image in missing:
            label = re.sub(
                r"\s+", " ", str(image.caption or image.alt or image.title or "")
            ).strip()
            full_marker = f"{image.marker}: {label}]" if label else f"{image.marker}]"
            anchor_span = None
            seen_labels = set()
            for candidate in (image.caption, image.alt, image.title):
                candidate = re.sub(r"\s+", " ", str(candidate or "")).strip()
                key = candidate.casefold()
                if not candidate or key in seen_labels:
                    continue
                seen_labels.add(key)
                pattern = re.compile(
                    r"\s+".join(re.escape(part) for part in candidate.split()),
                    re.IGNORECASE,
                )
                for match in pattern.finditer(text):
                    span = match.span()
                    if not any(span[0] < end and start < span[1] for start, end in claimed):
                        anchor_span = span
                        break
                if anchor_span is not None:
                    break
            if anchor_span is None and label:
                figure_number = re.match(r"(?i)^figure\s+(\d+)\s*[:.]", label)
                if figure_number:
                    number = re.escape(figure_number.group(1))
                    reference_pattern = re.compile(
                        rf"\bFigure[\s\xa0]+(?:【[^】\n]{{0,120}}?†)?"
                        rf"{number}(?:†[^】\n]*)?】?",
                        re.IGNORECASE,
                    )
                    for match in reference_pattern.finditer(text):
                        span = match.span()
                        if any(
                            span[0] < end and start < span[1]
                            for start, end in claimed
                        ):
                            continue
                        claimed.append(span)
                        replacements.append((span[0], span[0], full_marker + "\n"))
                        anchor_span = span
                        break
            if anchor_span is None:
                unanchored.append(full_marker)
                continue
            if not any(
                start == anchor_span[0] and end == anchor_span[0]
                for start, end, _replacement in replacements
            ):
                claimed.append(anchor_span)
                replacements.append((anchor_span[0], anchor_span[1], full_marker))

        for start, end, marker in sorted(replacements, reverse=True):
            text = f"{text[:start]}{marker}{text[end:]}"
        if unanchored:
            heading = "Images on page:"
            suffix = "\n".join(unanchored)
            text = f"{text.rstrip()}\n\n{heading}\n{suffix}" if text.strip() else f"{heading}\n{suffix}"
        return text

    def _process_html(self, body: str, url: str) -> BrowserPage:
        from lxml import etree
        from lxml import html as lxml_html
        import trafilatura

        body = re.sub(r"[\U00010000-\U0001FFFF]", "", body or "")
        try:
            root = lxml_html.fromstring(body)
        except Exception as exc:
            raise BrowserUsageError(f"Error parsing URL `{url}`: {exc}") from exc

        title_nodes = root.xpath("//title")
        title = (
            " ".join(title_nodes[0].itertext()).strip()
            if title_nodes
            else self._domain(url) or url
        )
        document_base = url
        base_nodes = root.xpath("//base[@href]")
        if base_nodes:
            candidate_base = urljoin(url, str(base_nodes[0].get("href") or "").strip())
            parsed_base = urlparse(candidate_base)
            if parsed_base.scheme.lower() in {"http", "https"} and parsed_base.netloc:
                document_base = candidate_base
        urls: Dict[str, str] = {}
        reverse: Dict[str, str] = {}
        current_domain = self._domain(url)
        for anchor in list(root.xpath(".//a[@href]")):
            href = str(anchor.get("href") or "")
            text = re.sub(r"\s+", " ", " ".join(anchor.itertext())).strip().replace("†", "‡")
            if not text or href.startswith(("mailto:", "javascript:")):
                continue
            if href.startswith("#"):
                self._replace_node(anchor, text)
                continue
            target = urljoin(document_base, href).replace("arxiv.org", "ar5iv.org")
            domain = self._domain(target)
            if not domain:
                continue
            link_id = reverse.get(target)
            if link_id is None:
                link_id = str(len(urls))
                urls[link_id] = target
                reverse[target] = link_id
            marker = (
                f"【{link_id}†{text}】"
                if domain == current_domain
                else f"【{link_id}†{text}†{domain}】"
            )
            self._replace_node(anchor, marker)

        images: List[PageImage] = []
        for index, image in enumerate(list(root.xpath(".//img"))):
            alt = re.sub(r"\s+", " ", str(image.get("alt") or "")).strip()
            image_title = re.sub(r"\s+", " ", str(image.get("title") or "")).strip()
            captions = image.xpath("ancestor::figure[1]//figcaption//text()")
            caption = re.sub(r"\s+", " ", " ".join(captions)).strip()
            source = ""
            srcset = str(image.get("srcset") or image.get("data-srcset") or "")
            if srcset:
                candidates = [
                    part.strip().split()
                    for part in srcset.split(",")
                    if part.strip()
                ]
                for candidate in reversed(candidates):
                    if candidate:
                        source = candidate[0]
                        break
            if not source:
                for attribute in ("data-original", "data-src", "data-lazy-src", "src"):
                    source = str(image.get(attribute) or "").strip()
                    if source:
                        break
            source = resolve_image_url(document_base, source)
            label = caption or alt or image_title
            replacement = f"[Image {index}: {label}]" if label else f"[Image {index}]"
            if source:
                images.append(
                    PageImage(
                        source_url=source,
                        page_url=url,
                        marker=f"[Image {index}",
                        original_name=image_name_from_url(source),
                        alt=alt,
                        caption=caption,
                        title=image_title,
                    )
                )
            self._replace_node(image, replacement)
        for node in list(root.xpath(".//script|.//style|.//noscript|.//svg|.//math")):
            parent = node.getparent()
            if parent is not None:
                parent.remove(node)

        clean_html = etree.tostring(root, encoding="unicode", method="html")
        with _HTML_PARSE_LOCK:
            text = trafilatura.extract(
                clean_html,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
                output_format="markdown",
            )
        if not text:
            text = re.sub(r"\s+", " ", root.text_content()).strip()
        text = self._restore_missing_image_markers(text, images)
        return BrowserPage(
            url=url,
            title=title,
            text=f"\nURL: {url}\n{text}",
            urls=urls,
            snippets=None,
            images=images,
        )

    def open(
        self,
        id: Union[int, str] = -1,
        cursor: int = -1,
        loc: int = -1,
        num_lines: int = -1,
        view_source: bool = False,
        web_multimodal_enabled: bool = False,
        page_image_open_enabled: bool = False,
        image_workdir: str = "",
        image_start: int = 1,
    ) -> str:
        snippet = None
        direct_url = False
        if isinstance(id, str) and id.strip().isdigit():
            id = int(id.strip())
        if isinstance(id, str):
            url = id
            direct_url = True
            current = None
        else:
            current = self._get_page(cursor)
            if id >= 0:
                url = current.urls.get(str(id))
                if url is None:
                    url = current.url
                snippet = (current.snippets or {}).get(str(id))
            else:
                url = current.url

        if not direct_url and isinstance(id, int) and id < 0 and not view_source:
            page = current
        elif not view_source and url in self.pages_by_url:
            page = self.pages_by_url[url]
        else:
            page = self.fetch_fn(url, view_source)
        self._add_page(page)

        if loc < 0:
            loc = 0
            if snippet is not None and snippet.line_idx is not None:
                loc = max(0, snippet.line_idx - 4)
        try:
            start, end = self._render_bounds(page, loc=loc, num_lines=num_lines)
            rendered = self._render(page, self.current_cursor, loc=loc, num_lines=num_lines)
            return self._with_page_images(
                rendered,
                page,
                [(start, end)],
                enabled=web_multimodal_enabled,
                open_enabled=page_image_open_enabled,
                workdir=image_workdir,
                image_start=image_start,
            )
        except Exception:
            self.pages.pop()
            raise

    def open_image(
        self,
        image_url: str,
        *,
        handle: str,
        caption: str = "",
        page_url: str = "",
        workdir: str = "",
    ) -> str:
        if not workdir:
            raise BrowserUsageError("browser.open image mode requires an image workdir.")
        image_url = str(image_url or "").strip()
        if not image_url:
            raise BrowserUsageError("browser.open image mode requires a source URL.")
        handle = str(handle or "page_img_1").strip()
        caption = re.sub(r"\s+", " ", html.unescape(str(caption or "Image"))).strip()
        image = PageImage(
            source_url=image_url,
            page_url=str(page_url or ""),
            marker="",
            original_name=image_name_from_url(image_url),
            alt=caption,
            caption=caption,
        )
        _, _, max_bytes = web_image_limits()
        attachment = materialize_image(
            image,
            workdir,
            handle,
            max_bytes=max_bytes,
            proxy_kwargs=S().proxy_kwargs,
        )
        marker = f"[{handle}: {caption or 'Image'}]"
        text = (
            f"Opened webpage image {marker}. Inspect the attached pixels in the "
            "next assistant turn."
        )
        return BrowserResult(text, [attachment])

    def find(
        self,
        pattern: str,
        cursor: int = -1,
        web_multimodal_enabled: bool = False,
        page_image_open_enabled: bool = False,
        image_workdir: str = "",
        image_start: int = 1,
    ) -> str:
        page = self._get_page(cursor)
        if page.snippets is not None:
            raise BrowserUsageError(
                "Cannot run `find` on search results page or find results page"
            )
        pattern = str(pattern or "").lower()
        if not pattern:
            raise BrowserUsageError("Find pattern cannot be empty.")
        lines = _strip_links("\n".join(_wrap_lines(page.text))).split("\n")
        chunks = []
        urls: Dict[str, str] = {}
        snippets: Dict[str, Extract] = {}
        match_ranges: List[Tuple[int, int]] = []
        line_idx = 0
        while line_idx < len(lines) and len(chunks) < 50:
            if pattern not in lines[line_idx].lower():
                line_idx += 1
                continue
            index = str(len(chunks))
            snippet = "\n".join(lines[line_idx : line_idx + 4])
            chunks.append(f"# 【{index}†match at L{line_idx}】\n{snippet}")
            urls[index] = page.url
            snippets[index] = Extract(page.url, snippet, f"#{index}", line_idx)
            match_ranges.append((max(0, line_idx - 4), line_idx + 8))
            line_idx += 4
        display = "\n\n".join(chunks) if chunks else f"No `find` results for pattern: `{pattern}`"
        result = BrowserPage(
            url=f"{page.url}/find?pattern={quote(pattern)}",
            title=f"Find results for text: `{pattern}` in `{page.title}`",
            text=display,
            urls=urls,
            snippets=snippets,
        )
        self._add_page(result)
        rendered = self._render(result, self.current_cursor)
        return self._with_page_images(
            rendered,
            page,
            match_ranges,
            enabled=web_multimodal_enabled,
            open_enabled=page_image_open_enabled,
            workdir=image_workdir,
            image_start=image_start,
        )

    @staticmethod
    def _page_image_positions(page: BrowserPage) -> List[Tuple[PageImage, int]]:
        lines = _wrap_lines(page.text)
        positions: List[Tuple[PageImage, int]] = []
        for image in page.images:
            for line_index, line in enumerate(lines):
                if image.marker and image.marker in line:
                    positions.append((image, line_index))
                    break
        return positions

    def _with_page_images(
        self,
        rendered: str,
        page: BrowserPage,
        ranges: Sequence[Tuple[int, int]],
        *,
        enabled: bool,
        open_enabled: bool,
        workdir: str,
        image_start: int,
    ) -> str:
        if not ranges:
            return rendered
        candidates: List[Tuple[PageImage, int]] = []
        seen_urls = set()
        exact_url_duplicates = 0
        analytics_rejected = 0
        for image, line_index in self._page_image_positions(page):
            if image.source_url in seen_urls:
                exact_url_duplicates += 1
                continue
            if _ANALYTICS_URL.search(image.source_url or ""):
                analytics_rejected += 1
                continue
            if any(start <= line_index < end for start, end in ranges):
                candidates.append((image, line_index))
                seen_urls.add(image.source_url)
        handle_by_url: Dict[str, str] = {}
        metadata_attachments: List[dict] = []
        if open_enabled:
            for image, line_index in candidates[:max(0, S().web_image_max_per_page)]:
                handle = f"page_img_{image_start + len(metadata_attachments)}"
                handle_by_url[image.source_url] = handle
                metadata_attachments.append({
                    "handle": handle,
                    "kind": "page_image",
                    "page_url": image.page_url or page.url,
                    "source_url": image.source_url,
                    "original_name": image.original_name,
                    "alt": image.alt,
                    "caption": image.caption or image.alt or image.title or "Image",
                    "title": image.title,
                    "line_start": line_index,
                    "line_end": line_index,
                })

        page_lines = _wrap_lines(page.text)

        def nearby_text(line_index: int, span: int = 3, limit: int = 110) -> str:
            lo = max(0, line_index - span)
            parts = []
            for i in range(lo, min(len(page_lines), line_index + span + 1)):
                if i == line_index:
                    continue
                parts.append(_prose_only(page_lines[i]))
            near = " ".join(p for p in parts if p)
            near = re.sub(r"\[?(?:page_img_\d+|Image \d+)[^\]]*\]?", " ", near)
            near = re.sub(r"images? on page[^.]*", " ", near, flags=re.I)
            near = re.sub(r"[\[\]]", " ", near)
            near = re.sub(r"\s+", " ", near).strip(" -–—·|")
            return near[:limit].rstrip() + ("…" if len(near) > limit else "")

        line_by_url = {image.source_url: idx for image, idx in candidates}

        def compact_marker(image: PageImage) -> str:
            caption = str(image.caption or image.alt or image.title or "Image").strip()
            handle = handle_by_url.get(image.source_url, "")
            label = caption or "Image"
            near = nearby_text(line_by_url.get(image.source_url, -1)) if image.source_url in line_by_url else ""
            if near:
                label = f"{label} · near: {near}"
            if handle:
                return f"[{handle}: {label}]"
            return f"[page image: {label}]"

        if not enabled and not open_enabled:
            return rendered

        fact_lines = []
        for image, _line_index in candidates:
            marker = compact_marker(image)
            if marker not in rendered:
                if image.marker and image.marker in rendered:
                    rendered = re.sub(
                        re.escape(image.marker) + r"[^\]]*\]",
                        lambda _m, _r=marker: _r,
                        rendered,
                        count=1,
                    )
                else:
                    fact_lines.append(f"- {marker}")
        if fact_lines:
            rendered += "\n\nImages on page (caption only; URLs kept out of context):\n" + "\n".join(fact_lines)

        if not enabled or not workdir:
            return BrowserResult(rendered, metadata_attachments) if metadata_attachments else rendered
        attachments: List[ImageAttachment] = []
        max_per_call, max_per_page, max_bytes = web_image_limits()
        attempted = 0
        failure_reasons: Dict[str, int] = {}
        for image, line_index in candidates[:max_per_page]:
            if len(attachments) >= min(max_per_call, max_per_page):
                break
            attempted += 1
            handle = f"page_img_{image_start + len(attachments)}"
            try:
                attachment = materialize_image(
                    image,
                    workdir,
                    handle,
                    line_start=line_index,
                    line_end=line_index,
                    max_bytes=max_bytes,
                    proxy_kwargs=S().proxy_kwargs,
                )
            except Exception as exc:
                message = str(exc).casefold()
                if "byte limit" in message or "exceeds" in message:
                    reason = "too_large"
                elif "too small" in message:
                    reason = "too_small"
                elif "format" in message or "svg" in message:
                    reason = "unsupported_format"
                elif "private" in message or "local" in message:
                    reason = "unsafe_or_nonpublic_url"
                elif "download" in message or "http" in message:
                    reason = "download_or_http"
                else:
                    reason = "invalid_or_corrupt"
                failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
                print(f"[browser:image] skipped {image.source_url}: {exc}")
                continue
            attachments.append(attachment)
        print("[browser:image-diagnostics] " + json.dumps({
            "url": page.url,
            "raw_image_mentions": len(page.images),
            "parsed_image_candidates": len(candidates) + exact_url_duplicates,
            "raw_image_candidates": len(candidates) + exact_url_duplicates,
            "parser_rejected": 0,
            "structured_fallback_used": False,
            "structured_candidates": 0,
            "semantic_rejected": 0,
            "analytics_rejected": analytics_rejected,
            "budget_truncated": max(
                0, len(candidates) - min(max_per_call, max_per_page)
            ),
            "download_attempted": attempted,
            "download_succeeded": len(attachments),
            "download_failed_by_reason": failure_reasons,
            "exact_duplicate_suppressed": exact_url_duplicates,
            "near_duplicate_suppressed": 0,
            "page_images_attached": len(attachments),
        }, ensure_ascii=False))
        return BrowserResult(rendered, attachments) if attachments else rendered

    def _render_bounds(
        self,
        page: BrowserPage,
        loc: int = 0,
        num_lines: int = -1,
    ) -> Tuple[int, int]:
        lines = _wrap_lines(page.text)
        if loc < 0:
            loc = 0
        elif loc >= len(lines):
            loc = max(0, len(lines) - 1)
        if num_lines > 0:
            return loc, min(loc + num_lines, len(lines))
        end = loc
        token_count = 0
        while end < len(lines):
            numbered = f"L{end}: {lines[end]}\n"
            next_count = token_count + _view_token_len(numbered)
            if end > loc and next_count > self.view_tokens:
                break
            token_count = next_count
            end += 1
        return loc, end

    def _render(
        self,
        page: BrowserPage,
        cursor: int,
        loc: int = 0,
        num_lines: int = -1,
    ) -> str:
        lines = _wrap_lines(page.text)
        loc, end = self._render_bounds(page, loc=loc, num_lines=num_lines)
        body = "\n".join(f"L{index}: {lines[index]}" for index in range(loc, end))
        title = page.title
        if page.url:
            title += f" ({unquote(page.url)})"
        return (
            f"[{cursor}] {title}\n"
            f"**viewing lines [{loc} - {end - 1}] of {len(lines) - 1}**\n\n"
            f"{body}"
        )


__all__ = [
    "BrowserPage",
    "BrowserResult",
    "BrowserSession",
    "BrowserUsageError",
    "Extract",
    "configure",
    "web_image_limits",
]
