"""Silero VAD — CPU 帧级流式人声检测

每个 30ms 音频帧（480 samples @ 16kHz）判断是否包含人声。
模型约 2MB，单帧推理 < 1ms，不占用 GPU。
"""

import torch


class VoiceDetector:

    def __init__(self, threshold: float = 0.5):
        self.model, _ = torch.hub.load(
            "snakers4/silero-vad", "silero_vad", onnx=True
        )
        self.threshold = threshold

    def is_speech(self, audio_chunk: torch.Tensor) -> bool:
        """判断一帧音频是否包含人声。

        Args:
            audio_chunk: float32 tensor, 16kHz mono, 长度 480 (30ms)
        """
        prob = self.model(audio_chunk, 16000).item()
        return prob > self.threshold

    def reset(self):
        """重置内部状态，在一段语音结束后调用。"""
        self.model.reset_states()
