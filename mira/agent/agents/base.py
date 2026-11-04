import copy
import hashlib
import json
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from qwen_agent.agents.fncall_agent import FnCallAgent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import Message
from qwen_agent.tools import BaseTool

from ..timing import TaskTiming, drain_tool_subtiming
from ..tools.common.runtime import web_image_features
from .react import (
    AnswerExtractionMixin,
    ContextMemoryMixin,
    EndpointMixin,
    ImageMessageMixin,
    PromptSchemaMixin,
    ToolCallParsingMixin,
    ToolResultBatch as ToolResultBatch,
    TrajectoryStatsMixin,
    TransportMixin,
    _normalize_tool_markup,
    _extract_tool_call_bodies,
    _tool_result_texts,
    capture_native_assistant,
)

_CFG = None


def _is_model_transport_failure(content) -> bool:
    return str(content or "").strip().lower().startswith("vllm server error!!!")


def _cfg():
    if _CFG is None:
        raise RuntimeError("configure_engine(cfg) not called (agents.base used before configure)")
    return _CFG


MAX_TURN = None
PYTHON_TOOL_ENABLED = None
PythonInterpreter = None
TOOL_CLASS = None
TOOL_MAP = None
KNOWN_TOOLS = None


def configure_engine(cfg) -> None:
    global _CFG, MAX_TURN, PYTHON_TOOL_ENABLED
    global PythonInterpreter, TOOL_CLASS, TOOL_MAP, KNOWN_TOOLS
    _CFG = cfg
    MAX_TURN = int(cfg.context.max_turn)

    PYTHON_TOOL_ENABLED = cfg.infra.enable_python_tool
    if PYTHON_TOOL_ENABLED:
        try:
            from ..tools.common import python as _python_tool
            from ..tools.common.python import PythonInterpreter as _PythonInterpreter
        except ImportError as exc:
            raise RuntimeError(
                "infra.enable_python_tool=true requires the optional Python tool dependencies. "
                "Install sandbox_fusion or set infra.enable_python_tool=false."
            ) from exc
        _python_tool.configure(cfg)
        PythonInterpreter = _PythonInterpreter
    else:
        PythonInterpreter = None

    TOOL_CLASS = []
    if PYTHON_TOOL_ENABLED:
        TOOL_CLASS.append(PythonInterpreter())
    TOOL_MAP = {tool.name: tool for tool in TOOL_CLASS}


    KNOWN_TOOLS = tuple(TOOL_MAP.keys()) + (
        "search", "visit", "PythonInterpreter",
        "text_search", "web_search", "local_search",
        "crop_and_search", "crop", "layout_parsing", "sharpen", "perspective_correct", "super_resolution", "image_search",
    )


def today_date():
    return date.today().strftime("%Y-%m-%d")


FORCE_ANSWER_PROMPT = (
    "Research is over. Tools are disabled.\n\n"
    "Give your best final answer now.\n"
    "Put only the final answer text between <answer> and </answer>."
)


class BaseReactAgent(
    TrajectoryStatsMixin,
    PromptSchemaMixin,
    ToolCallParsingMixin,
    ContextMemoryMixin,
    ImageMessageMixin,
    EndpointMixin,
    TransportMixin,
    AnswerExtractionMixin,
    FnCallAgent,
):
    discard_all_keeps_tool_history = False
    final_answer_reserve_s = 120.0
    tool_budget_s = {
        "search": 60.0,
        "visit": 60.0,
        "PythonInterpreter": 90.0,
    }

    def _research_deadline(self, task_deadline):
        if task_deadline is None:
            return None
        return task_deadline - self.final_answer_reserve_s

    def _tool_map(self) -> Dict:
        return TOOL_MAP or {}

    def _known_tools(self) -> Tuple:
        return KNOWN_TOOLS or ()

    def _forced_final_prompt(self, budget_sentence: str) -> str:
        return FORCE_ANSWER_PROMPT

    def _forced_final_turn(self, messages: List[Dict], tl, prompt: str) -> str:
        user_limit_msg = {"role": "user", "content": prompt}
        messages.append(user_limit_msg)
        tl.full_messages.append(copy.deepcopy(user_limit_msg))
        content = self.call_server(
            messages,
            max_tries=2,
            disable_tools=True,
            timeout_override_s=max(300.0, self.final_answer_reserve_s),
        )
        final_assistant_msg = {"role": "assistant", "content": content.strip()}
        messages.append(final_assistant_msg)
        tl.full_messages.append(copy.deepcopy(final_assistant_msg))
        return content

    def _forced_final_prediction(self, content) -> str:
        extracted = self._extract_answer_candidate(content)
        if extracted:
            return extracted
        raw = self._strip_tool_markup(str(content or "")).strip()
        if not raw or "vllm server error" in raw.lower():
            return ""
        if raw.lower() in {"answer not found.", "no answer found."}:
            return ""
        return raw[:16000]

    def _open_task_log_dir(self, data: Dict, tl, question: str) -> None:
        filename = data.get("filename", "")
        task_id = data.get("task_id", None)
        rollout_idx = data.get("rollout_idx", None)
        tl.current_filename = filename
        tl.image_refs = self._item_image_refs(data["item"])

        if filename:
            task_dir_name = filename
        elif task_id:
            task_dir_name = f"task_{task_id}"
        else:
            question_hash = hashlib.md5(question.encode("utf-8")).hexdigest()[:8]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_dir_name = f"task_{question_hash}_{timestamp}"

        parts = [self.base_log_dir, task_dir_name]
        if rollout_idx is not None:
            parts.append(f"iter{rollout_idx}")
        tl.task_log_dir = os.path.join(*parts)
        Path(tl.task_log_dir).mkdir(parents=True, exist_ok=True)

        suffix = "" if rollout_idx is None else f", rollout_idx: {rollout_idx}"
        print(
            f"[Memory] Task log directory created: {tl.task_log_dir} "
            f"(filename: {filename or 'N/A'}, task_id: {task_id or 'N/A'}{suffix})"
        )

    def _save_final_trajectory(self, messages: List[Dict], tl, *,
                               trigger_reason: str, round_num: int) -> None:
        if not (self.memory_enabled and tl.task_log_dir):
            return
        try:
            self._save_trajectory(
                messages,
                state=tl.memory_state,
                trigger_reason=trigger_reason,
                round_num=round_num,
                is_final=True,
            )
        except Exception as e:
            print(f"[Memory] Warning: Failed to save final trajectory: {e}")

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict, BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 cfg=None,
                 **kwargs):
        self.cfg = cfg if cfg is not None else _cfg()
        self.final_answer_reserve_s = 60.0 * max(
            0.0, float(self.cfg.context.final_answer_reserve_minutes)
        )
        self.web_multimodal_enabled = bool(
            self.cfg.infra.web_multimodal_enabled
        )
        self.web_image_features = web_image_features(self.cfg)
        self.image_workspace_enabled = self.web_image_features.image_workspace
        self.reverse_image_search_enabled = self.web_image_features.reverse_image_search
        self.reverse_image_pixel_feedback_enabled = self.web_image_features.reverse_image_pixels
        self.page_image_pixel_feedback_enabled = self.web_image_features.page_image_pixels
        self.page_image_open_enabled = self.web_image_features.page_image_open
        self.python_image_auto_feedback_enabled = (
            self.web_image_features.python_image_auto_feedback
        )
        self.tool_image_feedback_enabled = self.web_image_features.any_tool_pixels
        self.image_history_policy = str(
            self.cfg.infra.image_history_policy or "retain"
        ).strip().lower()
        if self.image_history_policy not in {"retain", "one_shot"}:
            raise ValueError(
                "infra.image_history_policy must be 'retain' or 'one_shot', got "
                f"{self.image_history_policy!r}"
            )
        self.llm_generate_cfg = llm["generate_cfg"]
        self.llm_local_path = llm["model_path"]
        requested_tool_names = {
            name for entry in (function_list or [])
            if (name := self._function_entry_name(entry))
        }
        self.python_tool_enabled = bool(
            PYTHON_TOOL_ENABLED and "PythonInterpreter" in requested_tool_names
        )
        self.function_list = function_list or []

        memory_threshold_str = self.cfg.context.threshold or '131072'
        self.base_memory_context_threshold = int(memory_threshold_str)
        print(f"Base memory threshold: {self.base_memory_context_threshold}")
        print(f"Python tool enabled: {self.python_tool_enabled}")
        self.memory_enabled = self.cfg.context.enabled
        self.memory_strategy = (
            self.cfg.context.strategy if self.cfg.context.strategy is not None else "discard_all"
        ).strip().lower()
        if self.memory_strategy != "discard_all":
            raise ValueError(
                f"Unsupported context strategy {self.memory_strategy!r}; "
                "only discard_all remains (condenser and hide_tool_result were removed)."
            )
        print(f"Memory strategy: {self.memory_strategy}")

        self.base_log_dir = self.cfg.infra.task_log_dir
        Path(self.base_log_dir).mkdir(parents=True, exist_ok=True)
        
        self._thread_local = threading.local()

        self._tokenizer = None
        self._model_token_counter = None
        self.round_timing_callback = None
    
    def _get_thread_state(self):
        if not hasattr(self._thread_local, 'initialized'):
            self._init_thread_state()
        return self._thread_local
    
    def _init_thread_state(self):
        tl = self._thread_local
        tl.initialized = True
        tl.native_assistant_message = None
        tl.memory_state = None
        tl.memory_context_threshold = self.base_memory_context_threshold
        tl.context_threshold_triggered = False
        tl.condenser_call_count = 0
        tl.full_messages = []
        tl.task_log_dir = None
        tl.current_question = None
        tl.current_filename = ""
        tl.trajectory_update_count = 0
        tl.model = None
        tl.preferred_endpoint = None
        tl.endpoint_failure_count = 0
        tl.user_prompt = None
        tl.image_refs = []
        tl.image_exposures = []
        tl.infer_timing = None
        tl.task_deadline = None






    def _postprocess_response(self, message, content):
        capture_native_assistant(self._get_thread_state(), message, content)
        return _normalize_tool_markup(content)

    def _refine_final_answer(self, content, messages):
        return content

    def _run(self, data: str, model: str, **kwargs) -> List[List[Message]]:
        task_started = time.perf_counter()
        self._init_thread_state()
        tl = self._thread_local
        tl.infer_timing = TaskTiming(started_at=task_started)
        drain_tool_subtiming()
        tl.model = model

        try:
            question = data["item"]["question"]
        except Exception:
            raw_msg = data["item"]["messages"][1]["content"]
            question = raw_msg.split("User:")[1].strip() if "User:" in raw_msg else raw_msg

        resume_messages = data.get("resume_messages")
        resume_num_rounds = int(data.get("resume_num_rounds") or 0)
        resume_enabled = isinstance(resume_messages, list) and len(resume_messages) > 0

        def has_extracted_answer(prediction) -> bool:
            prediction_value = str(prediction or "").strip()
            return bool(
                prediction_value
                and prediction_value.lower()
                not in {
                    "answer not found.",
                    "answer not found",
                    "no answer found.",
                    "no answer found",
                }
                and not prediction_value.lower().startswith("no answer found after ")
            )

        def attach_resume_metadata(result: Dict) -> Dict:
            result.setdefault("forced_submit", False)
            result.setdefault("forced_submit_reason", None)
            result["answer_extracted"] = has_extracted_answer(result.get("prediction"))
            _last_assistant = ""
            for _m in reversed(result.get("messages") or []):
                if _m.get("role") == "assistant":
                    _c = _m.get("content")
                    _last_assistant = _c if isinstance(_c, str) else str(_c)
                    break
            result["transport_failed"] = "vllm server error" in _last_assistant.lower()
            if resume_enabled:
                result["resumed_from_messages"] = True
                result["resume_start_num_rounds"] = resume_num_rounds
                result["resume_previous_termination"] = data.get("resume_previous_termination")
                result["resume_previous_prediction"] = data.get("resume_previous_prediction")
            messages_for_audit = result.get("messages") or []
            delivered_tool_schema = self._delivered_tool_schema_for_audit()
            result["delivered_tool_schema_channel"] = self.tool_schema_channel()
            result["delivered_tool_schema"] = delivered_tool_schema
            result["delivered_tool_schema_text"] = self._render_tool_schema_text_for_audit(
                delivered_tool_schema
            )
            result["system_prompt_with_tool_schema"] = (
                self._system_prompt_with_tool_schema_for_audit(messages_for_audit)
            )
            if tl.image_exposures:
                result["image_exposures"] = copy.deepcopy(tl.image_exposures)
            trace = tl.infer_timing
            trace.note("termination", result.get("termination"))
            trace.set_metric("agent.rounds", int(result.get("num_rounds") or 0))
            trace.add_stage("agent.total", time.perf_counter() - task_started)
            timing_snapshot = trace.snapshot()
            result["trajectory_stats"] = self._build_trajectory_stats(
                result.get("messages") or [],
                timing_snapshot,
            )
            result["_infer_timing"] = timing_snapshot
            return result

        def finalized(prediction, termination, *, finalization_mode=None,
                      forced_submit_reason=None) -> Dict:
            result = {
                "question": question,
                "messages": messages,
                "prediction": prediction,
                "termination": termination,
                "num_rounds": round,
                "task_log_dir": tl.task_log_dir,
                "filename": tl.current_filename,
            }
            if finalization_mode is not None:
                result["finalization_mode"] = finalization_mode
            if forced_submit_reason is not None:
                result["forced_submit"] = True
                result["forced_submit_reason"] = forced_submit_reason
            return attach_resume_metadata(result)

        tl.current_question = question
        task_id = data.get("task_id", None)
        self._open_task_log_dir(data, tl, question)

        max_minutes = int(self.cfg.context.max_runtime_minutes)
        task_deadline = (
            time.perf_counter() + max_minutes * 60
            if max_minutes > 0 else None
        )
        tl.task_deadline = task_deadline
        tl.user_prompt = question
        system_prompt = self._apply_tool_schema_channel(
            self.build_system_prompt(self.python_tool_enabled)
        )
        cur_date = today_date()
        if "DATE" in system_prompt:
            system_prompt = system_prompt.replace("DATE", cur_date, 1)
        else:
            system_prompt = system_prompt + str(cur_date)

        user_prompt = tl.user_prompt
        from ..prompts import MM_PROTOCOL_PROMPTS

        if resume_enabled:
            messages = copy.deepcopy(resume_messages)
            self._prepare_one_shot_resume(messages)
            print(
                f"[Resume] Continuing from saved messages: "
                f"message_count={len(messages)}, start_num_rounds={resume_num_rounds}, "
                f"previous_termination={data.get('resume_previous_termination')}"
            )
        else:
            if (self.cfg.prompt.system_prompt_name in MM_PROTOCOL_PROMPTS
                    and tl.image_refs):
                messages = [
                    {"role": "system", "content": system_prompt},
                    self._mm_protocol_user_message(user_prompt, tl.image_refs),
                ]
            else:
                if tl.image_refs:
                    raise ValueError(
                        f"system_prompt_name={self.cfg.prompt.system_prompt_name!r} has no multimodal "
                        f"message protocol but this item carries {len(tl.image_refs)} image(s)"
                    )
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
        tl.full_messages = copy.deepcopy(messages)
        num_llm_calls_available = max(0, MAX_TURN - resume_num_rounds)
        round = resume_num_rounds
        consecutive_bad_steps = 0
        tl.forced_finalize_reason = None
        fail_fast_invalid_trajectory = os.environ.get(
            "FAIL_FAST_INVALID_TRAJECTORY", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        
        while num_llm_calls_available > 0:
            remaining_s = (
                task_deadline - time.perf_counter()
                if task_deadline is not None else None
            )
            if remaining_s is not None and remaining_s <= self.final_answer_reserve_s:
                print(
                    f"Per-question wall-clock budget ({max_minutes}min) has "
                    f"{max(0.0, remaining_s):.1f}s left; reserving it for one final answer."
                )
                content = self._compact_final_answer_retry(
                    question,
                    messages,
                    tl,
                    f"the {max_minutes}-minute wall-clock limit was reached",
                    timeout_override_s=max(
                        1.0, min(self.final_answer_reserve_s, remaining_s)
                    ),
                    deadline_monotonic=task_deadline,
                )
                prediction = self._extract_tagged_answer(content)
                self._save_final_trajectory(
                    messages, tl, trigger_reason="timeout", round_num=round
                )
                if not prediction:
                    return finalized(
                        f'No answer found after {max_minutes}min',
                        f'No answer found after {max_minutes}min',
                        finalization_mode="empty",
                        forced_submit_reason="wall_clock_limit",
                    )
                return finalized(
                    prediction,
                    'generate an answer as wall-clock limit reached',
                    finalization_mode="compact_deadline",
                    forced_submit_reason="wall_clock_limit",
                )

            round += 1
            num_llm_calls_available -= 1
            round_started = time.perf_counter()
            round_checkpoint = tl.infer_timing.checkpoint()
            round_timing = {"round": round, "tools": []}

            def finish_round(*, final_answer=False, protocol_error=False):
                round_timing["final_answer"] = bool(final_answer)
                if protocol_error:
                    round_timing["protocol_error"] = True
                round_timing["round_total_s"] = float(
                    f"{time.perf_counter() - round_started:.6f}"
                )
                round_timing["trace_delta"] = tl.infer_timing.delta_since(round_checkpoint)
                tl.infer_timing.add_round(dict(round_timing))
                self._emit_round_timing(task_id, data.get("rollout_idx"), round_timing)

            tl.context_threshold_triggered = False
            visible_image_count = self._message_image_part_count(messages)
            round_timing["visible_image_count"] = visible_image_count
            _t_llm = time.time()
            content = self.call_server(
                messages,
                deadline_monotonic=self._research_deadline(task_deadline),
            )
            _llm_secs = time.time() - _t_llm
            round_timing["llm_s"] = float(f"{_llm_secs:.6f}")
            if _is_model_transport_failure(content):
                raise RuntimeError("model_transport_failure: actor/API request exhausted")
            actor_returned = bool(str(content or "").strip()) and not str(content).startswith((
                "vllm server error!!!",
                "content filter rejected!!!",
            ))
            if actor_returned and visible_image_count:
                exposure = self._consume_one_shot_images(
                    messages,
                    actor_call_id=round,
                )
                if exposure:
                    round_timing["consumed_image_count"] = exposure["visible_image_count"]
                    round_timing["visible_image_handles"] = exposure["visible_image_handles"]
            print(f'Round {round}: {content}')
            if '<tool_response>' in content:
                pos = content.find('<tool_response>')
                content = content[:pos]
            assistant_msg = self.build_assistant_response_message(content)
            messages.append(assistant_msg)
            tl.full_messages.append(copy.deepcopy(assistant_msg))

            has_complete_answer = bool(self._extract_tagged_answer(content))
            if not has_complete_answer:
                no_tool_call = (not _extract_tool_call_bodies(content)
                                and not self._has_tool_intent(content))
                if no_tool_call and not self._is_degenerate_repetition(content):
                    if self._extract_answer_candidate(content):
                        has_complete_answer = True
            degenerate_response = (
                not has_complete_answer and self._is_degenerate_repetition(content)
            )
            if degenerate_response:
                tl.degenerate_streak = getattr(tl, "degenerate_streak", 0) + 1
                if tl.degenerate_streak >= 3:
                    print(
                        f"[Finalization] Degenerate repetition x{tl.degenerate_streak} "
                        f"at round {round}; forcing an answer."
                    )
                    num_llm_calls_available = 0
                    tl.forced_finalize_reason = "degenerate_repetition"
                else:
                    print(
                        f"[Finalization] round {round}: possible repetition "
                        f"(warning {tl.degenerate_streak}/3); continuing."
                    )
                    tl.degenerate_warned = True
                    degenerate_response = False
            else:
                tl.degenerate_streak = 0



            if has_complete_answer:
                content = self._refine_final_answer(content, messages)
                finish_round(final_answer=True)
                break

            tool_call_bodies = _extract_tool_call_bodies(content)
            if not degenerate_response and tool_call_bodies:
                _g_name, _g_args, _g_err = self._normalize_tool_call(tool_call_bodies[-1].strip())
                _sig = (_g_name, json.dumps(_g_args, sort_keys=True, ensure_ascii=False)
                        if isinstance(_g_args, dict) else str(_g_args))
                if _g_err is None and _sig == getattr(tl, "last_tool_sig", None):
                    tl.same_call_streak = getattr(tl, "same_call_streak", 1) + 1
                else:
                    tl.same_call_streak = 1
                    tl.last_tool_sig = _sig
                if tl.same_call_streak >= 5:
                    print(f"[Finalization] identical tool call x{tl.same_call_streak} "
                          f"({_g_name}) at round {round}; forcing an answer.")
                    num_llm_calls_available = 0
                    tl.forced_finalize_reason = "identical_tool_call"
                    degenerate_response = True
            if not degenerate_response and tool_call_bodies:
                raw_tool_call = tool_call_bodies[-1].strip()
                tool_name, tool_args, tool_error = self._normalize_tool_call(raw_tool_call)


                _t_tool = time.time()
                if tool_error is not None:
                    result = tool_error
                else:
                    result = self.custom_call_tool(
                        tool_name,
                        tool_args,
                        task_deadline_monotonic=task_deadline,
                    )

                result_texts = _tool_result_texts(result)
                tool_failed = (
                    tool_error is not None
                    or any(
                        text.startswith("[Tool Error]")
                        or text.startswith("[Python Interpreter Error]")
                        or text.startswith("[PythonInterpreter Error]")
                        or text.startswith("[Json Parse Error]")
                        for text in result_texts
                    )
                )
                consecutive_bad_steps = consecutive_bad_steps + 1 if tool_failed else 0
                if consecutive_bad_steps >= 3:
                    print(
                        "[Finalization] Three consecutive tool/format errors; "
                        "forcing an answer instead of spending the remaining turns."
                    )
                    num_llm_calls_available = 0
                    tl.forced_finalize_reason = "tool_errors"

                tool_seconds = time.time() - _t_tool
                tool_label = tool_name if isinstance(tool_name, str) and tool_name else "invalid"
                detailed_timings = self.drain_tool_call_timings()
                if detailed_timings:
                    for event in detailed_timings:
                        event_name = event.get("name") or "invalid"
                        event_seconds = float(event.get("seconds") or 0.0)
                        event_subphases = event.get("subphases_s") or {}
                        tl.infer_timing.add_stage(f"tool.{event_name}", event_seconds)
                        tl.infer_timing.increment(f"tool.{event_name}.calls")
                        for subphase, seconds in event_subphases.items():
                            tl.infer_timing.add_stage(f"tool_subphase.{subphase}", seconds)
                        round_timing["tools"].append({
                            "name": event_name,
                            "seconds": float(f"{event_seconds:.6f}"),
                            "error": bool(event.get("error")),
                            "subphases_s": {
                                name: float(f"{value:.6f}")
                                for name, value in event_subphases.items()
                            },
                        })
                        print(
                            f"[timer] task={task_id} round={round} llm={_llm_secs:.1f}s "
                            f"tool={event_name}={event_seconds:.1f}s"
                        )
                else:
                    tl.infer_timing.add_stage(f"tool.{tool_label}", tool_seconds)
                    tl.infer_timing.increment(f"tool.{tool_label}.calls")
                    subtiming = drain_tool_subtiming()
                    for subphase, seconds in subtiming.items():
                        tl.infer_timing.add_stage(f"tool_subphase.{subphase}", seconds)
                    round_timing["tools"].append({
                        "name": tool_label,
                        "seconds": float(f"{tool_seconds:.6f}"),
                        "error": bool(tool_error is not None or tool_failed),
                        "subphases_s": {
                            name: float(f"{value:.6f}") for name, value in subtiming.items()
                        },
                    })
                    print(f"[timer] task={task_id} round={round} llm={_llm_secs:.1f}s tool={tool_name}={tool_seconds:.1f}s")
                for tool_msg in self.build_tool_response_messages(result):
                    messages.append(tool_msg)
                    tl.full_messages.append(copy.deepcopy(tool_msg))
                if getattr(tl, "degenerate_warned", False):
                    tl.degenerate_warned = False
                    warn = {"role": "user", "content": (
                        "Warning: your last turn repeated itself heavily. Do not resend the "
                        "same query or reasoning. Use what the tool just returned, change your "
                        "angle, or answer now with <answer></answer>."
                    )}
                    messages.append(warn)
                    tl.full_messages.append(copy.deepcopy(warn))
                if tool_failed and fail_fast_invalid_trajectory:
                    print(
                        "[Finalization] First tool/protocol error ends this attempt; "
                        "the caller may retry this request."
                    )
                    finish_round(protocol_error=True)
                    self._save_final_trajectory(
                        messages, tl, trigger_reason="tool_error", round_num=round
                    )
                    return finalized(
                        "No answer found.", "tool_error",
                        finalization_mode="invalid_trajectory",
                        forced_submit_reason="tool_error",
                    )
            elif not has_complete_answer and not degenerate_response:
                observation = (
                    "<tool_response>\n"
                    "Error: Invalid content format. Content must contain <tool_call> "
                    "or <answer> tags. Let's try again.\n"
                    "</tool_response>"
                )
                user_msg = {"role": "user", "content": observation}
                messages.append(user_msg)
                tl.full_messages.append(copy.deepcopy(user_msg))
                consecutive_bad_steps += 1
                if fail_fast_invalid_trajectory:
                    print(
                        "[Finalization] First invalid-format response ends this attempt; "
                        "the caller may retry this request."
                    )
                    finish_round(protocol_error=True)
                    self._save_final_trajectory(
                        messages, tl, trigger_reason="invalid_format", round_num=round
                    )
                    return finalized(
                        "No answer found.", "invalid_format",
                        finalization_mode="invalid_trajectory",
                        forced_submit_reason="invalid_format",
                    )
                if consecutive_bad_steps >= 3:
                    print(
                        "[Finalization] Three consecutive invalid-format responses; "
                        "forcing an answer."
                    )
                    num_llm_calls_available = 0
                    tl.forced_finalize_reason = "invalid_format"
            
            with self._timed_stage("tokenize.total"):
                token_count = self.count_tokens(messages)
            tl.infer_timing.max_metric("context.max_tokens", token_count)
            
            dynamic_threshold = tl.memory_context_threshold
            
            print(f"round: {round}, token count: {token_count}, dynamic threshold: {dynamic_threshold}")

            has_answer = False
            round_timing["context_tokens"] = token_count
            finish_round(final_answer=bool(self._extract_tagged_answer(content)))

            if self._extract_tagged_answer(content):
                has_answer = True

            if (not has_answer and
                token_count >= dynamic_threshold and 
                self.memory_enabled and 
                not tl.context_threshold_triggered):
                tl.infer_timing.increment("context.management_triggers")
                print(f"[Memory] CONTEXT_THRESHOLD trigger: token_count ({token_count}) >= dynamic threshold ({dynamic_threshold}, base: {self.base_memory_context_threshold})")
                original_token_count = token_count
                self._discard_all_memory_context(messages, round_num=round)
                token_count_after = self.count_tokens(messages)
                reduction = original_token_count - token_count_after
                reduction_percent = (reduction / original_token_count * 100) if original_token_count > 0 else 0
                print("[Memory] Discard-all reset completed, continuing from fresh context")
                print(f"[Memory] Token count reduced: {original_token_count} -> {token_count_after} (reduced by {reduction} tokens, {reduction_percent:.1f}%)")
                token_count = token_count_after
                tl.memory_context_threshold = self.base_memory_context_threshold
                tl.context_threshold_triggered = False


            if self._extract_tagged_answer(content):
                content = self._refine_final_answer(content, messages)
                break
            if num_llm_calls_available <= 0 and not self._extract_tagged_answer(content):
                print("LLM call count reached the limit, forcing a final answer.")
                content = self._forced_final_turn(
                    messages, tl,
                    self._forced_final_prompt(
                        "You have reached the maximum research budget for this question."
                    ),
                )
                if (self.cfg.context.force_answer_retry
                        and not self._extract_tagged_answer(content)):
                    content = self._compact_final_answer_retry(
                        question, messages, tl,
                        "the research budget was exhausted",
                        deadline_monotonic=task_deadline,
                    ) or content
                prediction = self._forced_final_prediction(content)
                self._save_final_trajectory(
                    messages, tl, trigger_reason="llm_call_limit", round_num=round
                )
                _force_why = getattr(tl, "forced_finalize_reason", None) or "llm_call_limit"
                _term = ('generate an answer as llm call limit reached'
                         if _force_why == "llm_call_limit"
                         else f'forced answer: {_force_why}')
                if not prediction:
                    return finalized(
                        'answer not found.', 'answer not found.',
                        finalization_mode="missing_extractable_answer",
                        forced_submit_reason=_force_why,
                    )
                return finalized(
                    prediction,
                    _term,
                    finalization_mode="tagged",
                    forced_submit_reason=_force_why,
                )


        last_content = messages[-1]['content']
        prediction = self._extract_tagged_answer(last_content)
        if not prediction:
            prediction = self._extract_answer_candidate(last_content)
        self._save_final_trajectory(
            messages, tl, trigger_reason="final", round_num=round
        )
        if prediction:
            return finalized(prediction, 'answer')
        return finalized(
            'No answer found.',
            'exceed available llm calls' if num_llm_calls_available == 0
            else 'answer not found',
        )

    def custom_call_tool(self, tool_name: str, tool_args: dict, **kwargs):
        if tool_name not in TOOL_MAP:
            return f"[Tool Error] Unknown tool: {tool_name}. Available tools: {list(TOOL_MAP.keys())}"

        if not isinstance(tool_args, dict):
            return f"[Tool Error] Invalid tool call format for {tool_name}: arguments must be object, got {tool_args!r}"

        task_deadline = kwargs.pop("task_deadline_monotonic", None)
        if task_deadline is None and hasattr(self, "_thread_local"):
            task_deadline = getattr(self._get_thread_state(), "task_deadline", None)
        component_cap = float(self.tool_budget_s.get(tool_name, 90.0))
        tool_deadline = time.perf_counter() + component_cap
        if task_deadline is not None:
            tool_deadline = min(
                tool_deadline,
                task_deadline - self.final_answer_reserve_s,
            )
        remaining_tool_s = tool_deadline - time.perf_counter()
        if remaining_tool_s <= 0:
            return (
                f"[Tool Error] {tool_name} skipped because the per-question "
                "deadline is reserved for the final answer."
            )
        kwargs["deadline_monotonic"] = tool_deadline
        if tool_name == "PythonInterpreter":
            kwargs["timeout"] = max(1, min(50, int(remaining_tool_s)))

        if tool_name == "PythonInterpreter" and not self.python_tool_enabled:
            return (
                "[Tool Error] PythonInterpreter is disabled for this run. "
                "Add PythonInterpreter to --tools and configure a live sandbox endpoint."
            )
        try:
            tool_args["params"] = tool_args
            raw_result = TOOL_MAP[tool_name].call(tool_args, **kwargs)
            return raw_result
        except Exception as e:
            return f"[Tool Error] {str(e)}"
