"""全局配置 — 所有可调参数集中管理"""

import torch


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = _detect_device()

# ── 服务 ──
HOST = "0.0.0.0"
PORT = 8765

# ── 音频 ──
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 32
FRAME_SIZE = 512  # Silero VAD 要求 16kHz 下最小 512 samples

# ── VAD ──
VAD_THRESHOLD = 0.5
SILENCE_TIMEOUT_MS = 500  # 静音超时 → 兜底提交尾部文本

# ── 流式 ASR ──
ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
ASR_GPU_MEMORY_UTILIZATION = 0.15
ASR_CHUNK_SIZE_SEC = 2.0
ASR_UNFIXED_CHUNK_NUM = 2
ASR_UNFIXED_TOKEN_NUM = 5
ASR_MAX_NEW_TOKENS = 32

# ── 翻译 (Qwen3-1.7B) ──
TRANSLATOR_MODEL = "Qwen/Qwen3-1.7B"
TRANSLATOR_GPU_MEMORY_UTILIZATION = 0.35
TRANSLATOR_MAX_TOKENS = 512

# ── TTS ──
TTS_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
TTS_SPEAKER = "Vivian"

# ── 语言映射 ──
LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}

# ── 标点集合 ──
# 句子标点 → 触发 ASR 提交翻译
SENTENCE_ENDINGS = frozenset("。！？.!?")
# 子句标点 → 在翻译输出侧触发 TTS 合成
CLAUSE_DELIMITERS = frozenset("，,、:：;；")
# TTS 侧切分用的全部分隔符
TTS_SPLIT_DELIMITERS = SENTENCE_ENDINGS | CLAUSE_DELIMITERS
