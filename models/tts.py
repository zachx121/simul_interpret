"""Qwen3-TTS — 子句级语音合成

使用 Qwen3-TTS-12Hz-1.7B-CustomVoice 模型，支持 10 种语言、9 种预设说话人。
Dual-Track 架构原生支持流式合成，首包延迟 97ms。

当前实现为子句级调用：每个子句独立调用 generate_custom_voice。
后续可升级为 faster-qwen3-tts 或 vLLM-Omni 的 token 级流式合成。
"""

import torch
import numpy as np
from qwen_tts import Qwen3TTSModel

from config import TTS_MODEL, TTS_SPEAKER, DEVICE


class TTSEngine:

    def __init__(self):
        dtype = torch.float32 if DEVICE == "cpu" else torch.bfloat16
        self.model = Qwen3TTSModel.from_pretrained(
            TTS_MODEL,
            device_map=DEVICE,
            dtype=dtype,
        )
        self.speaker = TTS_SPEAKER

    def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
        """合成单个文本片段。

        Args:
            text: 待合成文本（通常是一个子句）
            language: 语言名称（如 "Chinese", "English"）

        Returns:
            (audio_array, sample_rate) — audio 为 float32 numpy array
        """
        wavs, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=self.speaker,
        )
        return wavs[0], sr
