import argparse
import fcntl
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from mira.agent.agents import load_agent
from mira.agent.config import load_config
from mira.agent.config.loader import _collect_roots
from mira.agent.engine import configure_engine
from mira.agent.prompts import PROMPT_REGISTRY


def load_requests(path):
    path = Path(path).resolve()
    requests, ids = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - {"id", "question", "images"}:
            raise ValueError(f"Line {number}: expected id, question and optional images")
        request_id = row.get("id")
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", request_id):
            raise ValueError(f"Line {number}: id must use letters, digits, dots, underscores or hyphens")
        if request_id in ids:
            raise ValueError(f"Duplicate request id: {request_id}")
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"Line {number}: question must be a nonempty string")
        images = row.get("images", [])
        if not isinstance(images, list) or any(not isinstance(image, str) or not image.strip() for image in images):
            raise ValueError(f"Line {number}: images must be a list of paths or URLs")
        resolved = []
        for image in images:
            if image.startswith(("https://", "http://", "data:image/")):
                resolved.append(image)
                continue
            local = Path(image.removeprefix("file://")).expanduser()
            local = (path.parent / local).resolve()
            if not local.is_file():
                raise FileNotFoundError(f"Line {number}: image not found: {local}")
            resolved.append(str(local))
        requests.append({"id": request_id, "question": question, "images": resolved})
        ids.add(request_id)
    if not requests:
        raise ValueError("The input file contains no requests")
    return requests


def validate_config(cfg):
    if not cfg.model.name:
        raise ValueError("Set model.name in the configuration")
    base = urlsplit(cfg.model.api_base or "")
    if base.scheme not in {"http", "https"} or not base.hostname:
        raise ValueError("Set model.api_base to an HTTP(S) API URL")
    if base.username or base.password or base.query or base.fragment:
        raise ValueError("Put model credentials in configs/secrets.yaml")
    if cfg.prompt.system_prompt_name not in PROMPT_REGISTRY:
        raise ValueError(f"Unknown system prompt: {cfg.prompt.system_prompt_name}")
    if not Path(cfg.context.tokenizer_path, "tokenizer.json").is_file():
        raise ValueError("Set context.tokenizer_path to your model tokenizer directory")
    if int(cfg.context.max_turn) < 1 or int(cfg.context.threshold) < 1:
        raise ValueError("Context threshold and max_turn must be positive")
    enabled = {name.strip() for name in cfg.model_tools.split(",") if name.strip()}
    allowed = {"browser.search", "browser.open", "browser.find", "PythonInterpreter"}
    if enabled - allowed:
        raise ValueError(f"Unknown tools: {sorted(enabled - allowed)}")
    credentials = []
    if "browser.search" in enabled:
        credentials.append((cfg.search.serper_key_ids, "serper.key_ids"))
    if "browser.open" in enabled:
        credentials.append((cfg.search.jina_api_keys, "jina.api_keys"))
    for value, name in credentials:
        if not value or str(value).startswith("YOUR_"):
            raise ValueError(f"Set {name} in configs/secrets.yaml")
    if "PythonInterpreter" in enabled and not cfg.infra.enable_python_tool:
        raise ValueError("Enable infra.enable_python_tool or remove PythonInterpreter from model.tools")


def check_context_window(cfg):
    try:
        with httpx.Client(trust_env=False, timeout=10) as client:
            response = client.get(
                cfg.model.api_base.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {cfg.model.api_key or 'EMPTY'}"},
            )
            response.raise_for_status()
            entries = response.json().get("data", [])
    except (httpx.HTTPError, ValueError):
        print("The model service did not report its context window; continuing.", flush=True)
        return
    for entry in entries:
        if entry.get("id") == cfg.model.name and entry.get("max_model_len"):
            window = int(entry["max_model_len"])
            required = int(cfg.context.threshold) + int(cfg.model.llm_max_tokens or 0)
            if required > window:
                raise ValueError(f"Context threshold plus generation limit ({required}) exceeds the served window ({window})")


def create_agent(cfg):
    validate_config(cfg)
    configure_engine(cfg)
    tools = [name.strip() for name in cfg.model_tools.split(",") if name.strip()]
    if "PythonInterpreter" in tools:
        from mira.agent.tools.common.python import preflight_sandbox
        ok, detail = preflight_sandbox(timeout=60)
        if not ok:
            raise RuntimeError(f"Sandbox preflight failed: {detail}")
    generation = {
        "temperature": float(cfg.model.llm_temperature or 1.0),
        "top_p": float(cfg.model.llm_top_p or 0.95),
        "presence_penalty": float(cfg.model.llm_presence_penalty or 0.0),
    }
    return load_agent(cfg)(
        cfg=cfg, function_list=tools,
        llm={"model": cfg.model.name, "model_path": cfg.context.tokenizer_path,
             "generate_cfg": generation},
    )


def run_request(agent, request):
    request_id = request["id"]
    task = {"item": request, "task_id": request_id, "filename": request_id}
    try:
        result = agent._run(task, agent.cfg.model.name)
    except Exception as exc:
        result = {"question": request["question"], "prediction": "", "messages": [],
                  "termination": "error", "error": str(exc), "answer_extracted": False}
    result["id"] = request_id
    result["images"] = request["images"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Run MIRA on text and image requests.")
    parser.add_argument("--input", required=True, help="JSONL containing id, question and optional images")
    parser.add_argument("--config", default="configs/mira.yaml")
    parser.add_argument("--output", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-turn", type=int, default=0)
    args = parser.parse_args()
    if args.workers < 1 or args.max_turn < 0:
        parser.error("Workers must be positive and max-turn cannot be negative")
    requests = load_requests(args.input)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = Path(args.output or f"workspace/{stamp}").resolve()
    roots = dict(_collect_roots(os.environ), OUTPUT_DIR=str(output))
    cfg = load_config(args.config, roots=roots)
    if args.api_base:
        cfg = replace(cfg, model=replace(cfg.model, api_base=args.api_base.rstrip("/")))
    if args.max_turn:
        cfg = replace(cfg, context=replace(cfg.context, max_turn=str(args.max_turn)))
    validate_config(cfg)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("This output directory is already in use") from None
        path = output / "results.jsonl"
        if path.exists():
            raise FileExistsError("Choose a new output directory; results.jsonl already exists")
        check_context_window(cfg)
        agent = create_agent(cfg)
        failed = 0
        with path.open("x", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_request, agent, request) for request in requests]
            for future in as_completed(futures):
                result = future.result()
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                failed += bool(result.get("error") or result.get("transport_failed") or not result.get("answer_extracted"))
                print(f"{result['id']}: {result['termination']}", flush=True)
        print(f"Results: {path}", flush=True)
        if failed:
            raise SystemExit(f"{failed} request(s) did not produce an answer")


if __name__ == "__main__":
    main()
