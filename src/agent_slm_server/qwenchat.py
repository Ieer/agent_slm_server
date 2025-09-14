# 從 y26hkx_server.qwenchat 重命名至 agent_slm_server.qwenchat
import os
from typing import Any
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_threads = os.getenv("TORCH_NUM_THREADS")
if _threads and _threads.isdigit():
    try:
        torch.set_num_threads(int(_threads))
    except Exception:  # noqa: BLE001
        pass

@dataclass
class GenerationOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class QwenChatAPI:
    def __init__(
        self,
        model_path: str = os.getenv("MODEL_PATH", "./models/qwen/Qwen2.5-0.5B-Instruct"),
        device_map: str | None = "auto",
        dtype: str | None = os.getenv("TORCH_DTYPE", "auto"),
        compile_model: bool = os.getenv("COMPILE_MODEL", "0") in {"1", "true", "TRUE", "True"},
        quantization: str | None = os.getenv("MODEL_QUANTIZATION", None),
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = "auto",
        max_input_tokens: int = int(os.getenv("MAX_INPUT_TOKENS", "2048")),
        default_system_prompt: str | None = None,
        low_cpu_mem_usage: bool = os.getenv("LOW_CPU_MEM_USAGE", "1") in {"1", "true", "True"},
        global_seed: int | None = (int(os.getenv("GLOBAL_SEED", "0")) if os.getenv("GLOBAL_SEED") else None),
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        torch_dtype: Any = "auto"
        if isinstance(dtype, str) and dtype != "auto":
            dtype_map = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            torch_dtype = dtype_map.get(dtype.lower(), "auto")
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": low_cpu_mem_usage,
        }
        if quantization:
            try:
                from transformers import BitsAndBytesConfig
                if quantization == "4bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                elif quantization == "8bit":
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                        llm_int8_threshold=6.0
                    )
            except ImportError:  # pragma: no cover
                print("警告：需要安裝 bitsandbytes 才能啟用量化")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        if not torch.cuda.is_available() and hasattr(self.model, "config") and hasattr(self.model.config, "attn_implementation"):
            try:
                self.model.config.attn_implementation = "eager"  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        self.model.eval()
        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.generation_defaults = {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        self.tools = tools
        self.tool_choice = tool_choice
        self.max_input_tokens = max_input_tokens
        self.default_system_prompt = default_system_prompt
        self.global_seed = global_seed
        if self.global_seed is not None:
            torch.manual_seed(self.global_seed)

    def build_generation_kwargs(self,
                                max_new_tokens: int,
                                temperature: float,
                                top_p: float,
                                repetition_penalty: float,
                                do_sample: bool,
                                extra: dict[str, Any] | None = None) -> dict[str, Any]:
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
            "do_sample": do_sample,
        }
        if extra:
            kwargs.update(extra)
        return kwargs

    def _truncate_inputs(self, inputs: dict[str, torch.Tensor], max_input_tokens: int) -> dict[str, torch.Tensor]:
        if "input_ids" not in inputs:
            return inputs
        seq_len = inputs["input_ids"].shape[1]
        if seq_len <= max_input_tokens:
            return inputs
        k = max_input_tokens
        out: dict[str, torch.Tensor] = {}
        for name, tensor in inputs.items():
            if tensor.dim() == 2 and tensor.shape[1] >= seq_len:
                out[name] = tensor[:, -k:]
            else:
                out[name] = tensor
        return out

    def _build_inputs(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, tool_choice: Any | None = None) -> dict[str, torch.Tensor]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = self.tokenizer([text], return_tensors="pt")
        return enc

    def _truncate_messages_role_aware(self, messages: list[dict[str, Any]], max_input_tokens: int, reserve_system: bool = True) -> list[dict[str, Any]]:
        if max_input_tokens <= 0:
            return messages[-1:] if messages else []
        if not messages:
            return []
        system_msg = None
        body_msgs: list[dict[str, Any]] = []
        for m in messages:
            if reserve_system and m.get("role") == "system" and system_msg is None:
                system_msg = m
            else:
                body_msgs.append(m)
        kept_rev: list[dict[str, Any]] = []
        total_tokens = 0
        for m in reversed(body_msgs):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                token_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                need = len(token_ids)
            else:
                need = 0
            if total_tokens + need <= max_input_tokens:
                kept_rev.append(m)
                total_tokens += need
            else:
                if isinstance(content, str) and need > 0:
                    remain = max(0, max_input_tokens - total_tokens)
                    if remain > 16:
                        ratio = remain / need
                        cut_chars = max(1, int(len(content) * ratio))
                        m2 = dict(m)
                        m2["content"] = content[-cut_chars:]
                        kept_rev.append(m2)
                        total_tokens = max_input_tokens
                break
        kept = list(reversed(kept_rev))
        if system_msg is not None:
            return [system_msg] + kept
        return kept

    def chat_messages(self,
                      messages: list[dict[str, Any]],
                      max_new_tokens: int = 512,
                      temperature: float = 0.7,
                      top_p: float = 0.9,
                      repetition_penalty: float = 1.1,
                      seed: int | None = None,
                      tools: list[dict[str, Any]] | None = None,
                      tool_choice: Any | None = None,
                      max_input_tokens: int | None = None,
                      return_usage: bool = False,
                      stop: list[str] | None = None) -> Any:
        use_tools = tools if tools is not None else self.tools
        use_tool_choice = tool_choice if tool_choice is not None else self.tool_choice
        if self.default_system_prompt and (not messages or messages[0].get("role") != "system"):
            messages = [{"role": "system", "content": self.default_system_prompt}] + messages
        if max_input_tokens:
            messages = self._truncate_messages_role_aware(messages, max_input_tokens)
        inputs = self._build_inputs(messages, tools=use_tools, tool_choice=use_tool_choice)
        cap_tokens = max_input_tokens or self.max_input_tokens
        inputs = self._truncate_inputs(inputs, cap_tokens)
        if seed is not None:
            torch.manual_seed(seed)
        do_sample = temperature != 0
        with torch.inference_mode():
            extra = {"input_ids": inputs["input_ids"], "attention_mask": inputs.get("attention_mask")}
            generated_ids = self.model.generate(**self.build_generation_kwargs(max_new_tokens, temperature, top_p, repetition_penalty, do_sample, extra))  # type: ignore[attr-defined]
        new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if stop:
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    break
        if not return_usage:
            return text
        prompt_tokens = int(inputs["input_ids"].shape[1])
        completion_tokens = int(new_tokens.shape[0])
        total_tokens = prompt_tokens + completion_tokens
        return GenerationOutput(text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=total_tokens)

    def chat(self, prompt: str, **gen_kwargs) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat_messages(messages, **gen_kwargs)

if __name__ == "__main__":  # pragma: no cover
    api = QwenChatAPI()
    prompt1 = "從以上文本 {今天的日期為2025年8月25號}，輸出格式為：mm/dd/yyyy，只輸出日期"
    print("回复1:", api.chat(prompt1))
    prompt2 = "從以上文本 {今天的日期為2025年8月25號}，輸出格式為：yyyy-mm-dd，只輸出日期"
    print("回复2:", api.chat(prompt2))
