from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import fields
from typing import Any, Dict, Optional, Tuple

import yaml

from .schema import AgentConfig, GROUPS, LEAVES


class _StyledStr(str):
    style = ""

    def __new__(cls, value, style=""):
        obj = super().__new__(cls, value)
        obj.style = style or ""
        return obj


class _StrScalarLoader(yaml.SafeLoader):
    def construct_yaml_str(self, node):
        return _StyledStr(self.construct_scalar(node), style=getattr(node, "style", ""))


_StrScalarLoader.add_constructor("tag:yaml.org,2002:str",
                                 _StrScalarLoader.construct_yaml_str)


_DROP_RESOLVER_TAGS = (
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:timestamp",
)
_StrScalarLoader.yaml_implicit_resolvers = {
    ch: kept
    for ch, entries in yaml.SafeLoader.yaml_implicit_resolvers.items()
    for kept in ([(t, rx) for (t, rx) in entries if t not in _DROP_RESOLVER_TAGS],)
    if kept
}


_ALLOWED_ROOTS = (
    "MODELS_ROOT", "MIRA_SECRETS",
    "OUTPUT_DIR",
)

_SNAPSHOT_KEYS = ("RANK", "SLURM_PROCID", "LOCAL_RANK")



_VAR_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _interp(value: Any, roots: Dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {k: _interp(v, roots) for k, v in value.items()}
    if isinstance(value, list):
        return [_interp(v, roots) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value

    def sub(m):
        name = m.group(1)
        if name not in roots:
            raise KeyError(
                f"${{{name}}} used in config but not an allowed root "
                f"(allowed: {', '.join(sorted(roots))}). Set the corresponding environment variable."
            )
        return roots[name]

    return _VAR_RE.sub(sub, value)


def _collect_roots(environ: Dict[str, str]) -> Dict[str, str]:
    return {k: environ[k] for k in _ALLOWED_ROOTS if k in environ}






_IDENTITY_KEYS = ("model.variant", "model.tools")


_SECRET_YAML_NAME = "secrets.yaml"


_SECRET_YAML_MAP: Dict[str, Dict[str, str]] = {
    "jina": {
        "api_keys":    "search.jina_api_keys",
        "read_base":   "fetch.jina_read_base",
        "fetch_proxy": "proxy.visit_fetch_proxy",
    },
    "serper": {
        "key_ids": "search.serper_key_ids",
    },
    "llm": {
        "api_key": "model.api_key",
    },
}


def _read_secret_yaml(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _resolve_secret_path(roots: Dict[str, str], warn: bool = True) -> Optional[Path]:
    path = Path(roots["MIRA_SECRETS"]).expanduser() if roots.get("MIRA_SECRETS") else Path(__file__).resolve().parents[3] / "configs" / "secrets.yaml"
    if roots.get("MIRA_SECRETS") and not path.is_file():
        raise FileNotFoundError("MIRA_SECRETS does not point to a file")
    return path if path.is_file() else None




def _overlay_set(tree: dict, provenance: Dict[str, str], dotted: str, value: str) -> None:
    if dotted in provenance:
        return
    group, sub = dotted.split(".", 1)
    tree.setdefault(group, {})
    if not isinstance(tree[group], dict):
        return
    tree[group][sub] = value
    provenance[dotted] = "overlay:secret"


def _apply_secret_overlay(tree: dict, provenance: Dict[str, str],
                          roots: Dict[str, str], environ: Dict[str, str] = None) -> None:
    secret_path = _resolve_secret_path(roots)
    if secret_path is None:
        return
    secrets = _read_secret_yaml(secret_path)
    for block, leaf_map in _SECRET_YAML_MAP.items():
        body = secrets.get(block)
        body = body if isinstance(body, dict) else {}
        for leaf, dotted in leaf_map.items():
            if leaf in body and body[leaf] is not None:
                _overlay_set(tree, provenance, dotted, str(body[leaf]))


def _validate_coupled(tree: dict) -> Dict[str, str]:
    identity = {}
    for dotted in _IDENTITY_KEYS:
        group, sub = dotted.split(".", 1)
        if isinstance(tree.get(group), dict) and sub in tree[group]:
            value = tree[group].pop(sub)
            if value is not None:
                identity[sub] = str(value)
    return identity




def _load_one(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_StrScalarLoader) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def _flat_dotted(body: dict) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for group, sub_body in body.items():
        if group == "_extends" or not isinstance(sub_body, dict):
            continue
        for sub in sub_body:
            out[f"{group}.{sub}"] = ""
    return out


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if k == "_extends":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_extends(path: Path, root_dir: Path,
                     _stack: tuple = ()) -> Tuple[dict, Dict[str, str]]:
    path = path.resolve()
    if path in _stack:
        chain = " -> ".join(str(p) for p in (*_stack, path))
        raise ValueError(f"_extends cycle: {chain}")

    raw = _load_one(path)
    merged: dict = {}
    prov: Dict[str, str] = {}
    for parent_rel in raw.get("_extends", []) or []:
        parent = (path.parent / parent_rel).resolve()
        if not parent.is_file():
            raise FileNotFoundError(f"{path}: _extends target not found: {parent}")
        p_tree, p_prov = _resolve_extends(parent, root_dir, _stack + (path,))
        merged = _deep_merge(merged, p_tree)
        prov.update(p_prov)

    try:
        rel = str(path.relative_to(root_dir))
    except ValueError:
        rel = str(path)
    for dotted in _flat_dotted(raw):
        prov[dotted] = f"yaml:{rel}"

    return _deep_merge(merged, raw), prov


_QUOTED_STYLES = ('"', "'")


def _strict_input_form(dotted: str, kind: str, raw) -> None:
    style = getattr(raw, "style", "")
    if kind == "int":
        if isinstance(raw, str) and style in _QUOTED_STYLES:
            raise ValueError(
                f"field {dotted}: number must be bare ({str(raw)} not \"{str(raw)}\")")
    elif kind == "bool":
        if isinstance(raw, bool):
            return
        if isinstance(raw, str):
            if style in _QUOTED_STYLES:
                raise ValueError(
                    f"field {dotted}: bool must be true/false, got '{str(raw)}'")
            if str(raw) not in ("true", "false"):
                raise ValueError(
                    f"field {dotted}: bool must be true/false, got '{str(raw)}'")


def _build_group(group_name: str, group_cls, tree: dict, roots: Dict[str, str],
                 provenance: Dict[str, str], sources: Dict[str, str]):
    kwargs = {}
    body = tree.get(group_name, {}) or {}
    if not isinstance(body, dict):
        raise TypeError(f"group '{group_name}' must be a mapping")
    known = {f.name for f in fields(group_cls)}
    for key in body:
        if key not in known:
            raise KeyError(f"unknown config key: {group_name}.{key}")

    for f in fields(group_cls):
        dotted = f"{group_name}.{f.name}"
        leaf = LEAVES[dotted]
        if f.name in body and body[f.name] is not None:
            raw = _interp(body[f.name], roots)
            if not isinstance(raw, (str, int, float, bool, type(None))):
                raise TypeError(f"{dotted} must be a scalar, got {type(raw).__name__}")
            _strict_input_form(dotted, leaf.kind, raw)
            if leaf.parse is not None:
                raw = ("true" if raw else "false") if isinstance(raw, bool) else str(raw)
                val = leaf.parse(raw)
            else:
                if isinstance(raw, bool):
                    val = "true" if raw else "false"
                elif isinstance(raw, (int, float)):
                    val = str(raw)
                else:
                    val = raw
            sources[dotted] = provenance.get(dotted, "yaml")
        else:
            val = leaf.default
            sources[dotted] = "default"
        kwargs[f.name] = val
    return group_cls(**kwargs)


def load_config(yaml_path: str | os.PathLike,
                roots: Optional[Dict[str, str]] = None,
                environ: Optional[Dict[str, str]] = None) -> AgentConfig:
    environ = dict(os.environ if environ is None else environ)
    if roots is None:
        roots = _collect_roots(environ)

    root_dir = Path(yaml_path).resolve().parent
    tree, provenance = _resolve_extends(Path(yaml_path), root_dir)
    tree.pop("_extends", None)

    top_known = set(GROUPS) | {"_extends"}
    for key in tree:
        if key not in top_known:
            raise KeyError(f"unknown top-level config group: {key} (allowed: {sorted(GROUPS)})")

    _apply_secret_overlay(tree, provenance, roots, environ)



    identity = _validate_coupled(tree)

    sources: Dict[str, str] = {}
    group_objs = {
        name: _build_group(name, cls, tree, roots, provenance, sources)
        for name, cls in GROUPS.items()
    }

    snapshot = {k: environ[k] for k in _SNAPSHOT_KEYS if k in environ}
    return AgentConfig(
        _snapshot=snapshot,
        _sources=sources,
        _roots=roots,
        _identity=identity,
        **group_objs,
    )
