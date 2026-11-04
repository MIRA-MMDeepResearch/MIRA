import base64
import json
import os
import random
import re
import secrets
import time
from dataclasses import dataclass
from typing import List, Optional

import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from requests.exceptions import Timeout
from sandbox_fusion import RunCodeRequest, run_code

from .runtime import ConfigSlot, web_image_features

__all__ = ["PythonInterpreter", "configure", "configured_endpoints", "preflight_sandbox"]

_MM_TIMEOUT = 60
_PRELUDE = (
    "try:\n"
    "    import socket as _sk\n"
    "    _sk.setdefaulttimeout(15)\n"
    "except Exception:\n"
    "    pass\n"
)


@dataclass(frozen=True)
class _Settings:
    endpoints: tuple
    image_workspace_enabled: bool
    python_image_auto_feedback_enabled: bool


def _build(cfg) -> _Settings:
    features = web_image_features(cfg)
    raw = cfg.infra.sandbox_fusion_endpoints or cfg.infra.sandbox_fusion_endpoint or ""
    endpoints = tuple(dict.fromkeys(e.strip() for e in raw.split(",") if e.strip()))
    return _Settings(endpoints, features.image_workspace, features.python_image_auto_feedback)


_SLOT = ConfigSlot("python", _build)
configure = _SLOT.configure
S = _SLOT.get


def configured_endpoints() -> List[str]:
    return list(S().endpoints)


def mm_protocol_active() -> bool:
    return S().image_workspace_enabled


_VISUAL_DISPLAY_RE = re.compile(
    r"(?:"
    r"\b(?:plt|pyplot|matplotlib\.pyplot)\.show\s*\("
    r"|\b(?:plt|pyplot|matplotlib\.pyplot)\.imshow\s*\("
    r"|\bimshow\s*\("
    r"|\bdisplay\s*\("
    r"|\bIPython\.display\b"
    r")",
    re.IGNORECASE,
)


def _visual_display_in_code(code: str) -> bool:
    return bool(_VISUAL_DISPLAY_RE.search(str(code or "")))


class MMPythonResult(str):
    def __new__(cls, text: str, new_image_paths=()):
        obj = super().__new__(cls, text)
        obj.new_image_paths = list(new_image_paths)
        return obj


_MM_RUNNER = r'''
import base64, io, json, os, subprocess, sys
N = "{nonce}"
WORK = os.path.abspath("mm_" + N)
EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
CAP = 2 * 1024 * 1024

OUT = os.path.abspath("mm_out.json")
IMAGES = {{}}

def emit(obj):
    obj["data"] = IMAGES
    with open(OUT, "w") as f:
        json.dump(obj, f)

try:
    spec = json.load(open("mm_data.json"))
    os.makedirs(WORK, exist_ok=True)
    inject = spec.get("images", {{}})
    for ref in inject:
        with open(os.path.join(WORK, ref), "wb") as f:
            f.write(base64.b64decode(inject[ref]))
    code = base64.b64decode(spec["code"]).decode("utf-8")
    code_file = os.path.join(WORK, "._user_code.py")
    with open(code_file, "w") as f:
        f.write(code)
    before = {{}}
    for name in os.listdir(WORK):
        p = os.path.join(WORK, name)
        if os.path.isfile(p):
            before[name] = os.path.getmtime(p)
    timed_out = False
    out, err = "", ""
    def _txt(v):
        if v is None:
            return ""
        return v.decode("utf-8", "replace") if isinstance(v, bytes) else v
    try:
        proc = subprocess.run([sys.executable, code_file], cwd=WORK,
                              capture_output=True, text=True, errors="replace",
                              timeout={timeout})
        out, err = proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as te:
        timed_out = True
        out, err = _txt(te.stdout), _txt(te.stderr)
    injected = set(inject)
    registered = []
    for name in sorted(os.listdir(WORK)):
        p = os.path.join(WORK, name)
        if not os.path.isfile(p):
            continue
        if name in injected or name == "._user_code.py":
            continue
        if not name.lower().endswith(EXTS):
            continue
        if name in before and os.path.getmtime(p) <= before[name]:
            continue
        try:
            from PIL import Image
            with Image.open(p) as im:
                im.verify()
        except Exception:
            continue
        if len(registered) >= 4:
            continue
        registered.append(name)
    marshalled = []
    for name in registered:
        p = os.path.join(WORK, name)
        raw = open(p, "rb").read()
        if len(raw) > CAP:
            try:
                from PIL import Image
                with Image.open(p) as im:
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    while True:
                        w, h = im.size
                        im = im.resize((max(1, int(w * 0.7)), max(1, int(h * 0.7))))
                        buf = io.BytesIO()
                        im.save(buf, format="JPEG", quality=85)
                        raw = buf.getvalue()
                        if len(raw) <= CAP or min(im.size) <= 64:
                            break
            except Exception:
                raw = b""
        if not raw or len(raw) > CAP:
            continue
        IMAGES[name] = base64.b64encode(raw).decode()
        marshalled.append(name)
    emit({{
        "ok": True, "timeout": timed_out,
        "stdout": base64.b64encode(out[:6100].encode("utf-8")).decode(),
        "stderr": base64.b64encode(err[:2100].encode("utf-8")).decode(),
        "stdout_trunc": len(out) > 6000, "stderr_trunc": len(err) > 2000,
        "images": marshalled,
    }})
except Exception as exc:
    emit({{"ok": False, "error": str(exc)}})
'''


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode()


def _parse_mm_output(files: Optional[dict]) -> dict:
    raw = (files or {}).get("mm_out.json")
    if not raw:
        return {"ok": False, "error": "runner result file missing"}
    try:
        payload = json.loads(base64.b64decode(raw).decode("utf-8"))
    except Exception as exc:
        return {"ok": False, "error": f"result decode failed: {exc}"}
    if not payload.get("ok"):
        return {"ok": False, "error": f"runner error: {payload.get('error', '')}"}
    images = []
    for name, data in (payload.get("data") or {}).items():
        try:
            images.append((name, base64.b64decode(data)))
        except Exception:
            continue
    order = {n: i for i, n in enumerate(payload.get("images", []))}
    images = sorted([p for p in images if p[0] in order], key=lambda p: order[p[0]])
    return {
        "ok": True,
        "timed_out": bool(payload.get("timeout")),
        "stdout": base64.b64decode(payload.get("stdout", "")).decode("utf-8", "replace"),
        "stderr": base64.b64decode(payload.get("stderr", "")).decode("utf-8", "replace"),
        "stdout_truncated": bool(payload.get("stdout_trunc")),
        "stderr_truncated": bool(payload.get("stderr_trunc")),
        "new_images": images,
    }


def _run_mm(code: str, endpoints: List[str], mm_images: Optional[dict], mm_workdir: str, budget_s: float) -> MMPythonResult:
    inject = {}
    for ref, path in (mm_images or {}).items():
        try:
            with open(path, "rb") as fh:
                inject[ref] = fh.read()
        except Exception as exc:
            return MMPythonResult(f"[PythonInterpreter Error]: cannot read {ref}: {exc}")
    nonce = secrets.token_hex(8)
    spec = {"code": _b64(code.encode("utf-8")), "images": {ref: _b64(raw) for ref, raw in inject.items()}}
    request = RunCodeRequest(
        code=_MM_RUNNER.format(nonce=nonce, timeout=_MM_TIMEOUT),
        language="python",
        run_timeout=_MM_TIMEOUT + 60,
        files={"mm_data.json": _b64(json.dumps(spec).encode("utf-8"))},
        fetch_files=["mm_out.json"],
    )
    deadline = time.monotonic() + budget_s
    res, last_error = None, "no attempt"
    for _ in range(3):
        if time.monotonic() >= deadline:
            last_error = "sandbox call exceeded its time budget"
            break
        try:
            out = run_code(request, max_attempts=1, client_timeout=max(1, min(_MM_TIMEOUT + 90, deadline - time.monotonic())), endpoint=random.choice(endpoints))
            res = _parse_mm_output(out.files)
        except Exception as exc:
            res, last_error = None, str(exc)
            continue
        if res["ok"]:
            break
        last_error, res = res["error"], None
    if res is None:
        return MMPythonResult(f"[PythonInterpreter Error]: {last_error}")

    saved_lines, new_paths = [], []
    if res["new_images"] and mm_workdir:
        os.makedirs(mm_workdir, exist_ok=True)
    for fname, raw in res["new_images"]:
        if mm_images is None or not mm_workdir:
            break
        used = sum(1 for key in mm_images if str(key).startswith("py_img_"))
        ref = f"py_img_{used + 1}"
        suffix = ".jpg" if raw.startswith(b"\xff\xd8\xff") else os.path.splitext(fname)[1].lower()
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
            suffix = ".png"
        path = os.path.join(mm_workdir, f"{ref}{suffix}")
        with open(path, "wb") as fh:
            fh.write(raw)
        mm_images[ref] = path
        saved_lines.append(f"[image saved] {fname} is now available as {ref}")
        new_paths.append(path)

    feedback_paths = new_paths if S().python_image_auto_feedback_enabled else []
    if res["timed_out"]:
        parts = [f"[PythonInterpreter Error] TimeoutError: Execution timed out after {_MM_TIMEOUT}s."] + saved_lines
        return MMPythonResult("\n".join(parts), feedback_paths)
    parts = []
    if res["stdout"]:
        parts.append(f"stdout:\n{res['stdout'][:6000]}" + (" ...[truncated]" if res["stdout_truncated"] else ""))
    if res["stderr"]:
        parts.append(f"stderr:\n{res['stderr'][:2000]}" + (" ...[truncated]" if res["stderr_truncated"] else ""))
    parts.extend(saved_lines)
    return MMPythonResult("\n".join(parts) if parts else "Finished execution.", feedback_paths)


def _run_text(code: str, endpoint: str, timeout: int) -> str:
    result = run_code(RunCodeRequest(code=code, language="python", run_timeout=timeout),
                      max_attempts=1, client_timeout=timeout, endpoint=endpoint)
    parts = []
    if result.run_result.stdout:
        parts.append(f"stdout:\n{result.run_result.stdout}")
    if result.run_result.stderr:
        parts.append(f"stderr:\n{result.run_result.stderr}")
    if result.run_result.execution_time >= timeout - 1:
        parts.append("[PythonInterpreter Error] TimeoutError: Execution timed out.")
    text = "\n".join(parts)
    return text if text.strip() else "Finished execution."


def _extract(params) -> str:
    if isinstance(params, dict):
        code = params.get("code", "") or params.get("raw", "")
    elif isinstance(params, str):
        code = params
        try:
            parsed = json5.loads(params)
            if isinstance(parsed, dict):
                code = parsed.get("code", "") or parsed.get("raw", "") or code
        except Exception:
            pass
    else:
        code = extract_code(params)
    triple = re.search(r"```[^\n]*\n(.+?)```", code, re.DOTALL)
    return triple.group(1) if triple else code


def preflight_sandbox(timeout: int = 10) -> tuple:
    endpoints = configured_endpoints()
    if not endpoints:
        return False, "Set infra.sandbox_fusion_endpoint."
    failures = []
    code = "import PIL, numpy\nopen('probe.txt', 'w').write('ok')"
    request = RunCodeRequest(code=code, language="python", run_timeout=timeout,
                             fetch_files=["probe.txt"])
    for endpoint in endpoints:
        try:
            result = run_code(request, max_attempts=1, client_timeout=timeout + 5, endpoint=endpoint)
            if (result.files or {}).get("probe.txt") == _b64(b"ok"):
                return True, endpoint
            failures.append(endpoint + ": Pillow, NumPy or file transfer unavailable")
        except Exception as exc:
            failures.append(endpoint + ": " + type(exc).__name__)
    return False, "; ".join(failures)


@register_tool("PythonInterpreter", allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = (
        "Execute Python code in a sandboxed environment. Use this for calculations, "
        "data parsing, and processing files or images already available in the current "
        "trajectory. Do not use it as a web browser; retrieve URLs through search or visit. "
        "Use only preinstalled libraries; this tool is not for installing packages, "
        "system dependencies, or building a custom environment. The observation is "
        "stdout/stderr text unless the tool contract explicitly returns an image, so "
        "print the concrete values you need to observe. "
        "The sandbox is stateless: every call starts from a fresh environment and previous "
        "imports, variables, files, and installed packages are gone. Include all required "
        "code in the current call and print any value you need to observe."
    )

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Use print() for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    @property
    def function(self) -> dict:
        return {"name": self.name, "description": self.description, "parameters": self.parameters}

    def call(self, params, files=None, timeout=50, **kwargs) -> str:
        try:
            endpoints = configured_endpoints()
            if not endpoints:
                return "[Python Interpreter Error]: No sandbox_fusion endpoints configured."
            code = _extract(params)
            if not code or not code.strip():
                if mm_protocol_active():
                    return "[PythonInterpreter Error]: Empty code."
                return "[Python Interpreter Error]: Empty code."
            if _visual_display_in_code(code):
                return (
                    "[PythonInterpreter Error] Notebook/GUI image display calls such "
                    "as plt.show(), imshow(), or display() are disabled because Python "
                    "observations return stdout/stderr text, not visual pixels. For a "
                    "webpage image marker returned by visit, call visit(image='page_img_N', "
                    "goal=...) to attach that selected webpage image as pixels. Do not use "
                    "Python merely to redisplay the original question image. Use Python "
                    "only when the result can be printed as text/numbers or saved as a "
                    "file/handle for a later tool."
                )

            code = _PRELUDE + code
            if mm_protocol_active():
                deadline = kwargs.get("deadline_monotonic")
                remaining = deadline - time.perf_counter() if deadline is not None else 240.0
                if remaining <= 0:
                    return "[PythonInterpreter Error] Per-tool deadline reached before sandbox creation."
                return _run_mm(code, endpoints, kwargs.get("mm_images"), kwargs.get("mm_workdir") or "", remaining)
            last_error = "[Python Interpreter Error]: All attempts failed."
            for _ in range(5):
                endpoint = random.choice(endpoints)
                try:
                    return _run_text(code, endpoint, timeout)
                except Timeout:
                    last_error = f"[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}."
                except Exception as e:
                    last_error = f"[Python Interpreter Error]: {e} on endpoint {endpoint}"
            return last_error
        except Exception as e:
            return f"[Python Interpreter Error]: {e}"
