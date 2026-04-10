"""测试共享工具

- load_wav / save_wav: soundfile 绕开 torchcodec 依赖
- cer: 字符级错误率（归一化后 SequenceMatcher）
- iter_frames: 定长分帧，尾部零填充
- TimestampLog: wall-clock 事件记录
"""

import time
from difflib import SequenceMatcher

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000
FRAME_SIZE = 512  # 32ms @16k, 与 config.FRAME_SIZE 一致

# 去掉常见中英文标点和空白后再算 CER
_PUNCT_TABLE = str.maketrans(
    "", "", "，。！？,.!?;；:：、 \t\n\r\"'“”‘’()（）[]【】<>《》"
)


def load_wav(path: str) -> tuple[np.ndarray, int]:
    """读 wav → (float32 mono ndarray, sample_rate)"""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32, copy=False), sr


def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """float32 → PCM16 wav（不依赖 torchcodec）"""
    sf.write(path, audio.astype(np.float32, copy=False), sr, subtype="PCM_16")


def normalize_text(s: str) -> str:
    return s.translate(_PUNCT_TABLE)


def cer(ref: str, hyp: str) -> float:
    """字符级错误率：1 - SequenceMatcher.ratio()（归一化去标点后计算）"""
    r = normalize_text(ref)
    h = normalize_text(hyp)
    if not r:
        return 0.0 if not h else 1.0
    return 1.0 - SequenceMatcher(None, r, h).ratio()


def iter_frames(audio: np.ndarray, frame_size: int = FRAME_SIZE):
    """把一维音频切成定长帧；最后不足一帧时零填充到完整长度。"""
    n = len(audio)
    for i in range(0, n, frame_size):
        chunk = audio[i : i + frame_size]
        if len(chunk) < frame_size:
            pad = np.zeros(frame_size - len(chunk), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        yield chunk


class TimestampLog:
    """记录相对时间的事件日志。"""

    def __init__(self):
        self._t0 = time.time()
        self.events: list[tuple[float, str]] = []

    def mark(self, label: str) -> None:
        self.events.append((time.time() - self._t0, label))

    def dump(self, indent: str = "    ") -> str:
        return "\n".join(f"{indent}[{t:7.3f}s] {msg}" for t, msg in self.events)
