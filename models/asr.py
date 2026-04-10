"""Qwen3-ASR 封装 — 支持 vLLM 流式 / Transformers 批量 两种后端

CUDA 环境：使用 vLLM 后端，逐帧流式识别 + 标点驱动提交。
非 CUDA 环境（macOS MPS / CPU）：使用 Transformers 后端，攒够音频后批量识别。

核心设计（committed_text 指针机制）：
  committed_text : 已提交给翻译的文本，不可撤回
  当前识别文本   : vLLM 为 state.text，Transformers 为 self._current_text
  pending        : 当前识别文本 减去 committed_text 的部分

每次识别后，在 pending 中查找句子标点（。！？.!?）。
如果找到，截取到最后一个句子标点为止的文本返回，同时推进 committed_text。
"""

import numpy as np
from qwen_asr import Qwen3ASRModel

from config import (
    ASR_MODEL,
    ASR_GPU_MEMORY_UTILIZATION,
    ASR_MAX_NEW_TOKENS,
    ASR_CHUNK_SIZE_SEC,
    ASR_UNFIXED_CHUNK_NUM,
    ASR_UNFIXED_TOKEN_NUM,
    SENTENCE_ENDINGS,
    SAMPLE_RATE,
    DEVICE,
)

_USE_VLLM = DEVICE.startswith("cuda")


class StreamingASR:

    def __init__(self):
        if _USE_VLLM:
            self.model = Qwen3ASRModel.LLM(
                model=ASR_MODEL,
                gpu_memory_utilization=ASR_GPU_MEMORY_UTILIZATION,
                max_new_tokens=ASR_MAX_NEW_TOKENS,
            )
        else:
            self.model = Qwen3ASRModel.from_pretrained(
                ASR_MODEL,
                max_new_tokens=ASR_MAX_NEW_TOKENS,
                device_map=DEVICE,
            )

        self.state = None
        self.committed_text = ""
        # Transformers 后端：攒音频帧
        self._audio_buffer: list[np.ndarray] = []
        self._current_text = ""
        # 每积累多少秒做一次识别
        self._recognize_interval_sec = ASR_CHUNK_SIZE_SEC

    # ── 会话生命周期 ──

    def start_session(self):
        """开始一个新的识别会话。"""
        self.committed_text = ""
        self._audio_buffer.clear()
        self._current_text = ""

        if _USE_VLLM:
            self.state = self.model.init_streaming_state(
                unfixed_chunk_num=ASR_UNFIXED_CHUNK_NUM,
                unfixed_token_num=ASR_UNFIXED_TOKEN_NUM,
                chunk_size_sec=ASR_CHUNK_SIZE_SEC,
            )

    def finish_session(self) -> str | None:
        """结束会话，返回剩余未提交的尾部文本。"""
        if _USE_VLLM:
            self.model.finish_streaming_transcribe(self.state)
            full_text = self.state.text
            self.state = None
        else:
            # 把剩余音频全部识别
            if self._audio_buffer:
                self._run_batch_recognize()
            full_text = self._current_text

        remaining = full_text[len(self.committed_text):].strip()
        self.committed_text = ""
        self._audio_buffer.clear()
        self._current_text = ""
        return remaining if remaining else None

    # ── 帧级处理 ──

    def feed_audio(self, audio_chunk: np.ndarray) -> str | None:
        """喂入一帧音频，如果检测到句子标点则返回待翻译句子。

        vLLM 后端：逐帧流式识别 + 标点驱动提交。
        Transformers 后端：纯累积缓冲，永远返回 None。实际识别延迟到
          finish_session（由 server.py 的 VAD 静音超时触发）。
          原因：
            1) `_run_batch_recognize` 是对累积 buffer 的全量重识别，每次重
               算 O(N) 的前缀。若在 feed_audio 里周期性触发，总工作量是 O(N²)。
            2) `_extract` 的前缀假设（committed_text 是新 full_text 的前
               缀）在全量重识别下容易被同音字误识打破，导致字符索引错位、
               emit 出乱七八糟的碎片。
          所以把增量 emit 完全让给 vLLM 后端，Transformers 后端只做"整段
          识别"。
        """
        if _USE_VLLM:
            self.model.streaming_transcribe(audio_chunk, self.state)
            return self._try_extract_sentence_vllm()
        else:
            self._audio_buffer.append(audio_chunk)
            return None

    # ── 内部方法 ──

    def _run_batch_recognize(self):
        """Transformers 后端：将累积音频拼接后一次性识别。"""
        audio = np.concatenate(self._audio_buffer)
        results = self.model.transcribe([(audio, SAMPLE_RATE)])
        if results:
            self._current_text = results[0].text

    def _try_extract_sentence_vllm(self) -> str | None:
        full_text = self.state.text
        return self._extract(full_text)

    def _try_extract_sentence_transformers(self) -> str | None:
        return self._extract(self._current_text)

    def _extract(self, full_text: str) -> str | None:
        """在 pending 文本中查找句子标点，截取并返回已完成的句子。"""
        pending = full_text[len(self.committed_text):]

        last_pos = -1
        for i, ch in enumerate(pending):
            if ch in SENTENCE_ENDINGS:
                last_pos = i

        if last_pos < 0:
            return None

        sentence = pending[:last_pos + 1].strip()
        if not sentence:
            return None

        self.committed_text = full_text[:len(self.committed_text) + last_pos + 1]
        return sentence
