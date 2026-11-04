import json
import os
import random
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

import httpx
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError

from .images import _rewrite_file_images_to_data_urls
from .messages import _message_content_text

def _rate_limit_retry_after(error) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(str(raw))
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


_NO_TIMEOUT_FALLBACK_MINUTES = 15


class TransportMixin:
    def call_server(self, msgs, max_tries=10, *, disable_tools=False,
                    timeout_override_s=None, deadline_monotonic=None,
                    max_tokens_override=None):
        if deadline_monotonic is None:
            deadline_monotonic = getattr(self._get_thread_state(), "task_deadline", None)
        started = time.perf_counter()
        self._timing_increment("llm.calls")
        try:
            return self._call_server_impl(
                msgs,
                max_tries=max_tries,
                disable_tools=disable_tools,
                timeout_override_s=timeout_override_s,
                deadline_monotonic=deadline_monotonic,
                max_tokens_override=max_tokens_override,
            )
        finally:
            self._timing_add_stage("llm.call_total", time.perf_counter() - started)

    def _call_server_impl(self, msgs, max_tries=10, *, disable_tools=False,
                          timeout_override_s=None, deadline_monotonic=None,
                          max_tokens_override=None):
        openai_api_key = self.cfg.model.api_key or "EMPTY"
        FAILOVER_THRESHOLD = 2

        external_base = (self.cfg.model.api_base or "").strip()
        external_key = (self.cfg.model.api_key or "").strip() if self.cfg.model.api_key else ""
        external_mode = external_base.startswith(("https://", "http://"))
        api_protocol = (self.cfg.model.api_protocol or "openai").strip().lower()
        if api_protocol != "openai":
            raise ValueError(
                f"model.api_protocol must be openai, got {api_protocol!r}")

        tl = self._get_thread_state()
        max_tries = max(1, min(int(max_tries), int(self.cfg.model.llm_max_attempts)))
        base_sleep_time = 1
        last_error = None
        timeout_same_ep_used = 0
        for attempt in range(max_tries):
            if deadline_monotonic is not None:
                remaining_s = deadline_monotonic - time.perf_counter()
                if remaining_s <= 0:
                    print("--- Per-question deadline reached before actor attempt ---")
                    break
            else:
                remaining_s = None
            self._timing_increment("llm.attempts")
            if external_mode:
                hosts, ports, endpoint_source = [external_base], [0], f"external:{external_base}"
                all_endpoints = [(external_base, 0)]
                selected_host, selected_port = external_base, 0
                openai_api_base = external_base
                openai_api_key = external_key or "EMPTY"
            else:
                hosts, ports, endpoint_source = self._load_server_endpoints()
                all_endpoints = [
                    (host.rsplit(":", 1)[0], int(host.rsplit(":", 1)[1]))
                    if ":" in host else (host, port)
                    for host in hosts for port in (ports[:1] if ":" in host else ports)
                ]
                selected_host, selected_port = self._select_endpoint(all_endpoints)
                openai_api_base = f"http://{selected_host}:{selected_port}/v1"

            trace = self._active_timing()
            if trace is not None:
                trace.record_endpoint(selected_host, selected_port)
                trace.note("llm.endpoint_source", endpoint_source)

            print(f"--- Attempting to call the service, try {attempt + 1}/{max_tries} ---")
            print(f"--- Endpoint source: {endpoint_source} (hosts={hosts}, ports={ports}) ---")
            print(f"--- Using endpoint: {selected_host}:{selected_port} (preferred={tl.preferred_endpoint}, f_count={tl.endpoint_failure_count}) ---")

            call_failed = False
            timeout_failure = False
            timeout_failover_succeeded = False
            rate_limit_delay = None
            try:
                timeout_s = float(
                    timeout_override_s
                    if timeout_override_s is not None
                    else 60.0 * float(self.cfg.model.llm_call_timeout_minutes)
                )
                if remaining_s is not None:
                    timeout_s = (
                        remaining_s if timeout_s <= 0
                        else min(timeout_s, remaining_s)
                    )
                if timeout_s <= 0:
                    timeout_s = 60.0 * float(_NO_TIMEOUT_FALLBACK_MINUTES)
                timeout = timeout_s
                max_tokens = int(
                    max_tokens_override
                    if max_tokens_override is not None
                    else self.cfg.model.llm_max_tokens
                )
                client = OpenAI(
                    api_key=openai_api_key,
                    base_url=openai_api_base,
                    http_client=httpx.Client(trust_env=False, timeout=timeout),
                    timeout=timeout,
                    max_retries=int(self.cfg.model.llm_sdk_max_retries),
                )
                _extra = {}
                _top_k = self.cfg.model.llm_top_k
                if _top_k.strip():
                    _extra['top_k'] = int(_top_k)
                _ctk = (self.cfg.model.chat_template_kwargs or "").strip()
                if _ctk:
                    _extra['chat_template_kwargs'] = json.loads(_ctk)
                _top_p = (self.cfg.model.llm_top_p or "").strip()
                request = dict(
                    model=tl.model,
                    messages=msgs,
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=self.llm_generate_cfg.get('temperature', 0.6),
                    top_p=float(_top_p) if _top_p else self.llm_generate_cfg.get('top_p', 0.95),
                    logprobs=True,
                    max_tokens=max_tokens,
                    presence_penalty=self.llm_generate_cfg.get('presence_penalty', 1.1),
                    extra_body=_extra or None,
                )
                if max_tokens <= 0:
                    request.pop("max_tokens")
                reasoning_effort = self.cfg.prompt.reasoning_effort
                if reasoning_effort:
                    request["reasoning_effort"] = reasoning_effort
                if external_mode or self.cfg.model.inline_local_images:
                    request["messages"] = _rewrite_file_images_to_data_urls(msgs)
                api_tools = [] if disable_tools else self.delivered_api_tools()
                if api_tools and external_mode:
                    real_to_alias, _ = self.wire_tool_name_maps()
                    if real_to_alias:
                        api_tools = [
                            {**spec, "function": {**fn, "name": real_to_alias.get(fn.get("name"), fn.get("name"))}}
                            if (fn := spec.get("function") or {}) else spec
                            for spec in api_tools
                        ]
                if api_tools:
                    request["tools"] = api_tools
                    request["tool_choice"] = self.cfg.model.tool_choice or "auto"
                with self._timed_stage("llm.api_request"):
                    chat_response = client.chat.completions.create(**request)
                content = chat_response.choices[0].message.content
                content = self._postprocess_response(chat_response.choices[0].message, content)

                usage = getattr(chat_response, "usage", None)
                trace = self._active_timing()
                if trace is not None and usage is not None:
                    for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(attr) if isinstance(usage, dict) else getattr(usage, attr, None)
                        if isinstance(value, (int, float)):
                            trace.add_metric(f"llm.{attr}", value)

                if content and content.strip():
                    print(f"--- Service call successful, received a valid response from {selected_host}:{selected_port} ---")
                    tl.endpoint_failure_count = 0
                    self._timing_increment("llm.successes")
                    return content.strip()
                else:
                    print(f"Warning: Attempt {attempt + 1} received an empty response from {selected_host}:{selected_port}.")
                    call_failed = True

            except (APIError, APIConnectionError, APITimeoutError) as e:
                last_error = e
                print(f"Error: Attempt {attempt + 1} failed with an API or network error on {selected_host}:{selected_port}: {e}")
                timeout_failure = isinstance(e, APITimeoutError) or "request timed out" in str(e).lower()
                if "content_filter" in str(e):
                    print("--- Non-retryable: prompt rejected by content moderation; abandoning this call ---")
                    return "content filter rejected!!!"
                self._timing_increment("llm.errors")
                self._timing_note("llm.last_error", type(e).__name__)
                if getattr(e, "status_code", None) == 429:
                    rate_limit_delay = _rate_limit_retry_after(e)
                    delay_label = (
                        f"{rate_limit_delay:.1f}s"
                        if rate_limit_delay is not None else "not provided"
                    )
                    print(
                        "--- Rate limited (429): Retry-After="
                        f"{delay_label}; applying rate-limit backoff ---"
                    )
                call_failed = True
            except Exception as e:
                last_error = e
                print(f"Error: Attempt {attempt + 1} failed with an unexpected error on {selected_host}:{selected_port}: {e}")
                self._timing_increment("llm.errors")
                self._timing_note("llm.last_error", type(e).__name__)
                call_failed = True

            if call_failed:
                tl.endpoint_failure_count += 1
                print(f"--- Endpoint {selected_host}:{selected_port} failure count: {tl.endpoint_failure_count}/{FAILOVER_THRESHOLD} ---")

                should_failover = (
                    not external_mode
                    and (timeout_failure or tl.endpoint_failure_count >= FAILOVER_THRESHOLD)
                )
                if should_failover:
                    new_endpoint = self._failover_endpoint(
                        tl.preferred_endpoint, all_endpoints, hosts, ports
                    )
                    if new_endpoint:
                        old_endpoint = tl.preferred_endpoint
                        tl.preferred_endpoint = new_endpoint
                        tl.endpoint_failure_count = 0
                        timeout_failover_succeeded = bool(timeout_failure)
                        reason = "actor timeout" if timeout_failure else "failure threshold"
                        print(f"--- Failover ({reason}): switched from {old_endpoint} to {new_endpoint} ---")
                        if timeout_failure:
                            self._timing_increment("llm.timeout_failovers")
                    else:
                        print(f"--- Failover: no alternative endpoints available, continuing with {tl.preferred_endpoint} ---")
                        tl.endpoint_failure_count = 0

                if timeout_failure:
                    can_retry_same_ep = (
                        not timeout_failover_succeeded
                        and not external_mode
                        and timeout_same_ep_used < 1
                        and attempt < max_tries - 1
                    )
                    can_retry_after_failover = (
                        timeout_failover_succeeded
                        and attempt < max_tries - 1
                    )
                    if can_retry_after_failover:
                        print(
                            "--- Actor request timed out; retrying this logical "
                            "LLM call on the selected alternate endpoint. ---"
                        )
                    elif can_retry_same_ep:
                        timeout_same_ep_used += 1
                        print(
                            "--- Actor request timed out; single endpoint -- "
                            "one same-endpoint retry before giving up. ---"
                        )
                    else:
                        print(
                            "--- Actor request timed out; no unused failover "
                            "attempt is available, stopping this logical call. ---"
                        )
                        break

            if attempt < max_tries - 1:
                sleep_time = base_sleep_time * (2 ** attempt) + random.uniform(0, 1)
                if rate_limit_delay is not None:
                    sleep_time = max(sleep_time, rate_limit_delay + random.uniform(0, 1))
                    sleep_time = min(sleep_time, 300)
                else:
                    sleep_time = min(sleep_time, 30)
                if deadline_monotonic is not None:
                    remaining_s = deadline_monotonic - time.perf_counter()
                    if remaining_s <= 0:
                        print("--- Per-question deadline reached; skipping actor retry ---")
                        break
                    sleep_time = min(sleep_time, remaining_s)
                
                print(f"Retrying in {sleep_time:.2f} seconds...")
                with self._timed_stage("llm.retry_sleep"):
                    time.sleep(sleep_time)
            else:
                print("Error: All retry attempts have been exhausted. The call has failed.")
                print(f"  last_error: {type(last_error).__name__}: {last_error}")
                print(f"  endpoint  : {selected_host}:{selected_port}  attempts={max_tries}")

        detail = f"{type(last_error).__name__}: {last_error}" if last_error else "no exception recorded"
        if external_mode:
            raise RuntimeError(
                f"external actor call failed after {max_tries} attempts: {last_error}"
            ) from last_error
        self._timing_note("llm.last_error", detail[:500])
        return (f"vllm server error!!! [{selected_host}:{selected_port} "
                f"after {max_tries} attempts] {detail[:400]}")

    def count_tokens(self, messages):
        if getattr(self, "_model_token_counter", None) is None:
            from ...token_counting import get_model_token_counter

            model_tokenizer_path = self.cfg.context.tokenizer_path or (None)
            try:
                self._model_token_counter = get_model_token_counter(
                    model_tokenizer_path
                )
            except (FileNotFoundError, ValueError):
                self._model_token_counter = False

        image_parts = 0
        prompt_lines = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                image_parts += sum(
                    1 for part in content
                    if isinstance(part, dict)
                    and part.get("type") in {"image_url", "input_image", "image"}
                )
            prompt_lines.append(
                f"{message.get('role', '')}: {_message_content_text(content)}"
            )
        full_prompt = "\n".join(prompt_lines)
        if self._model_token_counter:
            total, text_tokens, exact_image_tokens = (
                self._model_token_counter.count_messages(
                    messages, tools=self.delivered_api_tools()
                )
            )
            print(
                "[context:tokens] "
                f"prompt={total}, text_and_template={text_tokens}, "
                f"images={exact_image_tokens}, image_count={image_parts}"
            )
            return total

        if self._tokenizer is None:
            from tokenizers import ByteLevelBPETokenizer
            count_tok = self.cfg.model.count_tokenizer_path
            self._tokenizer = ByteLevelBPETokenizer(
                os.path.join(count_tok, "vocab.json"),
                os.path.join(count_tok, "merges.txt"),
            )
        return len(self._tokenizer.encode(full_prompt).ids) + 1024 * image_parts
