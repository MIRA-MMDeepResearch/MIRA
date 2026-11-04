import copy
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional


class ContextMemoryMixin:
    def _save_memory_log(self, original_messages: List[Dict], compressed_messages: List[Dict], 
                        state: Dict, trigger_reason: str, round_num: int):
        try:
            tl = self._get_thread_state()
            tl.condenser_call_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            log_data = {
                "condenser_call_number": tl.condenser_call_count,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "state": state,
                "original_message_count": len(original_messages),
                "compressed_message_count": len(compressed_messages),
                "messages_removed": len(original_messages) - len(compressed_messages)
            }
            
            log_file = os.path.join(tl.task_log_dir, f"condenser_call_{tl.condenser_call_count}_{timestamp}.json")
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)
            
            print(f"[Memory] Log saved to: {log_file}")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save memory log: {e}")
            import traceback
            traceback.print_exc()
    
    def _remove_prev_state_from_messages(self, messages: List[Dict]) -> List[Dict]:
        cleaned_messages = copy.deepcopy(messages)
        
        for msg in cleaned_messages:
            if msg.get("role") == "user" and "content" in msg:
                content = msg["content"]
                if not isinstance(content, str):
                    continue
                prev_state_start = content.find("====================\nRESEARCH STATE SUMMARY")
                if prev_state_start != -1:
                    msg["content"] = content[:prev_state_start].rstrip()
        
        return cleaned_messages

    def _save_trajectory(self, messages: List[Dict], state: Optional[Dict] = None,
                                         trigger_reason: str = "unknown", round_num: int = 0,
                                         is_final: bool = False,
                                         no_memory_messages: Optional[List[Dict]] = None):
        try:
            tl = self._get_thread_state()
            
            if tl.current_question is None:
                print("[Memory] Warning: current_question is None, skipping trajectory save")
                return
            
            if tl.task_log_dir is None:
                print("[Memory] Warning: task_log_dir is None, skipping trajectory save")
                return
            
            if not is_final:
                tl.trajectory_update_count += 1
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            delivered_tool_schema = self._delivered_tool_schema_for_audit()
            delivered_tool_schema_text = self._render_tool_schema_text_for_audit(
                delivered_tool_schema
            )
            
            trajectory_data = {
                "question": tl.current_question,
                "is_final": is_final,
                "condenser_update_id": tl.trajectory_update_count if not is_final else None,
                "condenser_call_id": tl.condenser_call_count if not is_final else None,
                "timestamp": timestamp,
                "datetime": datetime.now().isoformat(),
                "round": round_num,
                "trigger_reason": trigger_reason,
                "prev_state": state,
                "messages": copy.deepcopy(messages),
                "message_count": len(messages),
                "delivered_tool_schema_channel": self.tool_schema_channel(),
                "delivered_tool_schema": copy.deepcopy(delivered_tool_schema),
                "delivered_tool_schema_text": delivered_tool_schema_text,
                "system_prompt_with_tool_schema": self._system_prompt_with_tool_schema_for_audit(messages),
                "image_exposures": copy.deepcopy(getattr(tl, "image_exposures", [])),
                "token_count": self.count_tokens(messages) if hasattr(self, 'count_tokens') else None
            }
            
            trajectory_file = os.path.join(tl.task_log_dir, "trajectories.jsonl")
            with open(trajectory_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(trajectory_data, ensure_ascii=False) + "\n")
            
            should_save_no_memory = is_final or no_memory_messages is not None
            if should_save_no_memory:
                if no_memory_messages is not None:
                    messages_no_memory = copy.deepcopy(no_memory_messages)
                else:
                    full_msgs = tl.full_messages if tl.full_messages else messages
                    messages_no_memory = self._remove_prev_state_from_messages(full_msgs)
                trajectory_data_no_memory = {
                    "question": tl.current_question,
                    "is_final": is_final,
                    "condenser_update_id": tl.trajectory_update_count if not is_final else None,
                    "condenser_call_id": tl.condenser_call_count if not is_final else None,
                    "timestamp": timestamp,
                    "datetime": datetime.now().isoformat(),
                    "round": round_num,
                    "trigger_reason": trigger_reason,
                    "prev_state": None,
                    "messages": messages_no_memory,
                    "message_count": len(messages_no_memory),
                    "delivered_tool_schema_channel": self.tool_schema_channel(),
                    "delivered_tool_schema": copy.deepcopy(delivered_tool_schema),
                    "delivered_tool_schema_text": delivered_tool_schema_text,
                    "system_prompt_with_tool_schema": self._system_prompt_with_tool_schema_for_audit(messages_no_memory),
                    "image_exposures": copy.deepcopy(getattr(tl, "image_exposures", [])),
                    "token_count": self.count_tokens(messages_no_memory) if hasattr(self, 'count_tokens') else None
                }
                
                trajectory_file_no_memory = os.path.join(tl.task_log_dir, "trajectories_no_memory.jsonl")
                with open(trajectory_file_no_memory, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(trajectory_data_no_memory, ensure_ascii=False) + "\n")
            
            if is_final:
                print(f"[Memory] Final trajectory saved (round {round_num}, trigger: {trigger_reason})")
            else:
                print(f"[Memory] Trajectory saved (update #{tl.trajectory_update_count}, round {round_num}, trigger: {trigger_reason})")
            
        except Exception as e:
            print(f"[Memory] Warning: Failed to save trajectory: {e}")
            import traceback
            traceback.print_exc()
    
    def _discard_all_memory_context(self, messages: List[Dict],
                                   original_messages: List[Dict] = None,
                                   trigger_reason: str = "CONTEXT_THRESHOLD_DISCARD_ALL",
                                   round_num: int = 0):
        if original_messages is None:
            original_messages = copy.deepcopy(messages)

        tl = self._get_thread_state()
        tl.memory_state = None

        first_user_idx = None
        for i, msg in enumerate(messages):
            if msg.get("role") == "user":
                first_user_idx = i
                break

        if first_user_idx is None:
            print("[Memory] Warning: No user message found to reset")
            return

        first_user_content = messages[first_user_idx]["content"]
        if "RESEARCH STATE SUMMARY (prev_state)" in first_user_content:
            prev_state_start = first_user_content.find("====================\nRESEARCH STATE SUMMARY")
            if prev_state_start != -1:
                first_user_content = first_user_content[:prev_state_start].rstrip()
                messages[first_user_idx]["content"] = first_user_content

        if self.discard_all_keeps_tool_history:
            think_re = re.compile(r'<think>.*?</think>\s*', re.S)
            stripped = 0
            for message in messages:
                if message.get("role") == "assistant" and "<think>" in (message.get("content") or ""):
                    content = think_re.sub('', message["content"])
                    if content != message["content"]:
                        message["content"] = content
                        stripped += 1
            print(f"[Memory] {trigger_reason}: stripped reasoning from {stripped} assistant msg(s), kept tool history")
        else:
            removed_count = len(messages) - first_user_idx - 1
            messages[:] = messages[:first_user_idx + 1]
            print(f"[Memory] {trigger_reason}: removed {removed_count} subsequent messages")

        compressed_messages = copy.deepcopy(messages)
        discard_state = {
            "strategy": "discard_all",
            "discarded": True,
            "threshold": self.base_memory_context_threshold,
        }
        self._save_memory_log(original_messages, compressed_messages, discard_state, trigger_reason, round_num)
        self._save_trajectory(
            original_messages,
            None,
            trigger_reason,
            round_num,
            no_memory_messages=copy.deepcopy(tl.full_messages if tl.full_messages else original_messages),
        )
