from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, NamedTuple, Optional


def _strip(v: str) -> str:            return v.strip()
def _strip_lower(v: str) -> str:      return v.strip().lower()
def _rstrip_slash(v: str) -> str:     return v.rstrip("/")
def _true(v: str) -> bool:            return v.lower() == "true"
def _true_stripped(v: str) -> bool:   return v.strip().lower() == "true"
def _on(v: str) -> bool:              return v.strip().lower() not in ("0", "false", "no", "off")
def _yes(v: str) -> bool:             return v.lower() in ("1", "true", "yes")
def _yes_cs(v: str) -> bool:          return v.strip() in ("1", "true", "True", "yes")

_BOOL_PARSERS = (_true, _true_stripped, _on, _yes, _yes_cs)


class Leaf(NamedTuple):
    parse: Optional[Callable[[str], object]]
    default: object
    secret: bool = False
    kind: str = "str"


def _L(default, parse=None, secret=False, kind="str") -> Leaf:
    parsed = default if (parse is None or default is None) else parse(default)
    if parse in _BOOL_PARSERS:
        kind = "bool"
    return Leaf(parse=parse, default=parsed, secret=secret, kind=kind)


LEAVES: Dict[str, Leaf] = {
    "model.name":                 _L(None),
    "model.llm_call_timeout_minutes": _L("4", kind="int"),
    "model.llm_max_attempts":     _L("10", kind="int"),
    "model.llm_sdk_max_retries":  _L("0", kind="int"),
    "model.llm_max_tokens":       _L("10000", kind="int"),
    "model.llm_top_k":            _L("", kind="int"),
    "model.llm_top_p":            _L(""),
    "model.llm_temperature":      _L(""),
    "model.llm_presence_penalty": _L(""),
    "model.chat_template_kwargs": _L(""),
    "model.count_tokenizer_path": _L("Qwen/Qwen3-0.6B"),
    "model.api_key":              _L(None, secret=True),
    "model.api_base":             _L(None),
    "model.api_protocol":         _L("openai", _strip_lower),
    "model.tool_schema_channel":  _L("", _strip_lower),
    "model.tool_schema_file":     _L("", _strip),
    "model.tool_choice":          _L("auto", _strip_lower),
    "model.tool_result_role":     _L("", _strip_lower),
    "model.inline_local_images":  _L("false", _true_stripped),

    "context.strategy":           _L(None),
    "context.enabled":            _L("true", _true),
    "context.threshold":          _L("131072", kind="int"),
    "context.tokenizer_path":     _L(""),

    "prompt.system_prompt_name":  _L("", _strip),
    "prompt.reasoning_effort":    _L("", _strip_lower),

    "context.max_turn":           _L("100", kind="int"),
    "context.max_runtime_minutes": _L("999999", kind="int"),
    "context.final_answer_reserve_minutes": _L("2", kind="int"),
    "context.force_answer_retry": _L("", _yes_cs),

    "search.serper_key_ids":      _L(None, secret=True),
    "search.jina_api_keys":       _L("", secret=True),
    "search.result_snippet_tokens": _L("512", kind="int"),
    "search.view_tokens":         _L("1024", kind="int"),

    "fetch.timeout":              _L("20", kind="int"),
    "fetch.tier_timeout":         _L("15", kind="int"),
    "fetch.pdf_max_pages":        _L("50", kind="int"),
    "fetch.playwright_concurrency":_L("4", kind="int"),
    "fetch.jina_read_base":       _L("https://r.jina.ai", _rstrip_slash),


    "cache.dir":                  _L(None),
    "cache.global_dir":           _L(None),
    "cache.tool_enabled":         _L("true", _true),
    "cache.tool_read":            _L("true", _true),
    "cache.tool_write":           _L("true", _true),
    "cache.tool_shards":          _L("32", kind="int"),
    "cache.tool_blob_max_bytes":  _L("4194304", kind="int"),
    "cache.tool_mem_max_bytes":   _L("268435456", kind="int"),
    "cache.tool_queue_max_bytes": _L("268435456", kind="int"),

    "infra.hostname_list":        _L("localhost"),
    "infra.ports":                _L("8000"),
    "infra.server_endpoints_file": _L("", _strip),
    "infra.task_log_dir":         _L("./task_logs"),
    "infra.enable_python_tool":   _L("true", _on),
    "infra.web_multimodal_enabled": _L("false", _true_stripped),
    "infra.image_workspace_enabled": _L(None, _true_stripped),
    "infra.reverse_image_search_enabled": _L(None, _true_stripped),
    "infra.reverse_image_pixel_feedback_enabled": _L(None, _true_stripped),
    "infra.page_image_pixel_feedback_enabled": _L(None, _true_stripped),
    "infra.page_image_open_enabled": _L(None, _true_stripped),
    "infra.python_image_auto_feedback_enabled": _L(None, _true_stripped),
    "infra.web_page_images_enabled": _L("true", _on),
    "infra.web_image_max_per_call": _L("4", kind="int"),
    "infra.web_image_max_per_page": _L("4", kind="int"),
    "infra.web_image_max_bytes":    _L(str(8 * 1024 * 1024), kind="int"),
    "infra.web_image_max_edge":     _L("1024", kind="int"),
    "infra.web_image_max_pixels":   _L(str(1024 * 1024), kind="int"),
    "infra.web_image_output_max_bytes": _L(str(1024 * 1024), kind="int"),
    "infra.web_image_jpeg_quality": _L("75", kind="int"),
    "infra.image_history_policy":   _L("retain", _strip_lower),
    "infra.sandbox_fusion_endpoints": _L(""),
    "infra.sandbox_fusion_endpoint":  _L(""),

    "proxy.visit_fetch_proxy":    _L(None, secret=True),
    "proxy.https_proxy":          _L(None, secret=True),
    "proxy.http_proxy":           _L(None, secret=True),

    "image.lens_url":             _L("https://google.serper.dev/lens", _rstrip_slash),
    "image.max_results":          _L("5", kind="int"),
    "image.lens_upload_endpoint": _L("https://catbox.moe/user/api.php"),
    "image.lens_timeout":         _L("45", kind="int"),
    "image.upload_timeout":       _L("60", kind="int"),
    "image.upload_concurrency":   _L("8", kind="int"),
}


def _leaf_annotation(leaf: Leaf):
    if leaf.parse in _BOOL_PARSERS:
        return bool
    if leaf.default is None:
        return Optional[str]
    return str


def _mk(group: str):
    subs = [k.split(".", 1)[1] for k in LEAVES if k.startswith(group + ".")]
    ns: Dict[str, object] = {"__annotations__": {}}
    for sub in subs:
        leaf = LEAVES[f"{group}.{sub}"]
        ns["__annotations__"][sub] = _leaf_annotation(leaf)
        ns[sub] = field(default=leaf.default)
    return dataclass(frozen=True)(type(f"_{group.capitalize()}Cfg", (), ns))


GROUP_NAMES = ("model", "context", "prompt", "search", "fetch",
               "cache", "infra", "proxy", "image")
GROUPS: Dict[str, type] = {g: _mk(g) for g in GROUP_NAMES}


@dataclass(frozen=True)
class AgentConfig:
    model:   object
    context: object
    prompt:  object
    search:  object
    fetch:   object
    cache:   object
    infra:   object
    proxy:   object
    image:   object
    _snapshot: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _sources:  Dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _roots:    Dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    _identity: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def raw(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._snapshot.get(name, default)

    @property
    def model_variant(self) -> str:
        return self._identity.get("variant", "")

    @property
    def model_tools(self) -> str:
        return self._identity.get("tools", "")

    def identity(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._identity.get(name, default)

    def source(self, dotted: str) -> str:
        return self._sources.get(dotted, "default")

    def dump(self) -> None:
        for dotted in sorted(LEAVES):
            group, sub = dotted.split(".", 1)
            leaf = LEAVES[dotted]
            src = self._sources.get(dotted, "default")
            if leaf.secret:
                val = getattr(getattr(self, group), sub)
                shown = ("set(len=%d)" % len(val)) if val else "unset"
            else:
                shown = repr(getattr(getattr(self, group), sub))
            print("[config] %s = %s  (%s)" % (dotted, shown, src))
