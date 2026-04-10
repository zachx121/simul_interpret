"""翻译模块 — 支持 vLLM / Transformers 两种后端

CUDA 环境：vLLM 加速推理。
非 CUDA 环境：Transformers + TextIteratorStreamer 流式输出。
"""

import torch
from config import (
    TRANSLATOR_MODEL,
    TRANSLATOR_GPU_MEMORY_UTILIZATION,
    TRANSLATOR_MAX_TOKENS,
    DEVICE,
)

_USE_VLLM = DEVICE.startswith("cuda")


class StreamingTranslator:

    def __init__(self):
        if _USE_VLLM:
            from vllm import LLM, SamplingParams
            self.llm = LLM(
                model=TRANSLATOR_MODEL,
                dtype="bfloat16",
                gpu_memory_utilization=TRANSLATOR_GPU_MEMORY_UTILIZATION,
                limit_mm_per_prompt={"image": 0, "video": 0},
            )
            self.params = SamplingParams(
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                max_tokens=TRANSLATOR_MAX_TOKENS,
            )
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
            self.tokenizer = AutoTokenizer.from_pretrained(TRANSLATOR_MODEL)
            self.model = AutoModelForCausalLM.from_pretrained(
                TRANSLATOR_MODEL,
                dtype=dtype,
                device_map=DEVICE,
            )

    def translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """同步翻译（一次性返回完整结果）。"""
        if _USE_VLLM:
            prompt = self._build_prompt(text, src_lang, tgt_lang)
            outputs = self.llm.generate([prompt], self.params)
            return self._clean_output(outputs[0].outputs[0].text)
        else:
            return self._translate_transformers(text, src_lang, tgt_lang)

    def translate_streaming(self, text: str, src_lang: str, tgt_lang: str):
        """流式翻译，yield 逐步累积的翻译文本。"""
        if _USE_VLLM:
            prompt = self._build_prompt(text, src_lang, tgt_lang)
            for output in self.llm.generate([prompt], self.params, use_tqdm=False):
                yield self._clean_output(output.outputs[0].text)
        else:
            yield from self._translate_streaming_transformers(text, src_lang, tgt_lang)

    def _translate_transformers(self, text: str, src_lang: str, tgt_lang: str) -> str:
        messages = self._build_messages(text, src_lang, tgt_lang)
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=TRANSLATOR_MAX_TOKENS,
                temperature=0.3,
                top_p=0.8,
                top_k=20,
                do_sample=True,
            )
        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self._clean_output(self.tokenizer.decode(new_tokens, skip_special_tokens=True))

    def _translate_streaming_transformers(self, text: str, src_lang: str, tgt_lang: str):
        from transformers import TextIteratorStreamer
        import threading

        messages = self._build_messages(text, src_lang, tgt_lang)
        input_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=TRANSLATOR_MAX_TOKENS,
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            do_sample=True,
            streamer=streamer,
        )

        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        accumulated = ""
        for chunk in streamer:
            accumulated += chunk
            yield self._clean_output(accumulated)

        thread.join()

    def _build_prompt(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """vLLM 后端使用的纯文本 prompt。"""
        return (
            f"Translate the following {src_lang} text to {tgt_lang}. "
            f"Output ONLY the translation, nothing else.\n\n"
            f"{text}"
        )

    def _build_messages(self, text: str, src_lang: str, tgt_lang: str) -> list[dict]:
        """Transformers 后端使用的 chat messages。"""
        return [
            {"role": "system", "content": (
                f"/no_think\n"
                f"You are a translator. Translate {src_lang} to {tgt_lang}. "
                f"Output ONLY the translation, nothing else."
            )},
            {"role": "user", "content": text},
        ]

    def _clean_output(self, text: str) -> str:
        """去除可能的 <think>...</think> 内容（Qwen3.5 thinking mode 兜底）。"""
        if "<think>" in text:
            idx = text.find("</think>")
            if idx != -1:
                text = text[idx + len("</think>"):]
        return text.strip()
