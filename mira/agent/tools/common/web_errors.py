import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_SECRET_QUERY_KEY = re.compile(
    r"(?i)(?:api[-_]?key|access[-_]?token|auth|authorization|credential|"
    r"password|secret|signature|sig|token)$"
)

_TARGET_PATTERNS = (
    ("http_403", re.compile(
        r"(?i)(?:http\s*403|403\s*(?:forbidden|error)|"
        r"status(?:_code)?\s*[=:]?\s*403|returned (?:an )?error\s*403)")),
    ("cloudflare_challenge", re.compile(
        r"(?i)(?:cf-ray|title:\s*just a moment|just a moment\.\.\.|"
        r"cloudflare[^\n]{0,100}(?:challenge|blocked|error|attention required|ray id))")),
    ("captcha", re.compile(
        r"(?i)(?:(?:requiring|requires?|complete|solve|warning:[^\n]{0,60})\s+"
        r"(?:a\s+)?captcha|captcha\s+(?:challenge|verification|required)|"
        r"verify (?:you are|that you are) human|human verification)")),
    ("robots_denied", re.compile(
        r"(?i)(?:disallow(?:ed)? by robots|robots\.txt[^\n]{0,80}(?:blocked|denied))")),
    ("access_denied", re.compile(
        r"(?i)(?:access\s+denied|request\s+blocked|blocked due to|"
        r"automated (?:access|requests)|bot protection|anti[- ]?bot)")),
    ("http_429", re.compile(
        r"(?i)(?:http\s*429|429\s*(?:too many requests|error)|"
        r"status(?:_code)?\s*[=:]?\s*429)")),
    ("login_required", re.compile(
        r"(?i)(?:sign\s+in\s+to\s+(?:continue|view|access|read)|"
        r"log\s+in\s+to\s+(?:continue|view|access|read)|"
        r"(?:authentication|login)\s+(?:is\s+)?required|"
        r"you\s+must\s+be\s+(?:signed|logged)\s+in)")),
    ("cookies_required", re.compile(
        r"(?i)(?:cookies?\s+(?:are\s+)?required|"
        r"(?:enable|allow)\s+(?:browser\s+)?cookies?\s+to\s+(?:continue|view|access)|"
        r"cannot\s+continue\s+without\s+cookies?)")),
    ("target_error_page", re.compile(
        r"(?im)^\s*(?:title:\s*)?(?:404\s+(?:not\s+found|error)|page\s+not\s+found|"
        r"500\s+(?:internal\s+server\s+error|error)|502\s+bad\s+gateway|"
        r"503\s+service\s+unavailable|504\s+gateway\s+timeout)\s*$")),
    ("fetch_error", re.compile(
        r"(?im)^\s*\[visit\]\s*error\b|^\s*error\s+fetching\s+(?:url|page)\b")),
)

_PROVIDER_PATTERNS = (
    ("provider_quota", re.compile(
        r"(?i)(?:http\s*402|402\s*(?:payment|required|error)|"
        r"insufficientbalance|account balance not enough|please recharge)")),
    ("provider_auth", re.compile(
        r"(?i)(?:invalid api key|unauthorized api|provider authentication failed)")),
)


def sanitize_url(url: str) -> str:
    raw = str(url or "").strip().rstrip(".,;:!?}')")
    if not raw.lower().startswith(("http://", "https://")):
        return raw[:2000]
    try:
        parts = urlsplit(raw)
        query = []
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            query.append((key, "<redacted>" if _SECRET_QUERY_KEY.search(key) else value))
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, urlencode(query), ""))[:4000]
    except Exception:
        return raw[:2000]


def is_web_failure(text: str = "", status: Optional[int] = None, *, scope: str = "target") -> bool:
    try:
        if status is not None and int(status) >= 400:
            return True
    except (TypeError, ValueError):
        pass
    body = str(text or "")
    return any(pattern.search(body) for _, pattern in (*_TARGET_PATTERNS, *_PROVIDER_PATTERNS))
