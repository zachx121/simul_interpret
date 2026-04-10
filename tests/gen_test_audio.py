"""生成固定文本的测试音频，用于后续 ASR/流式管线测试。

用法:
    python tests/gen_test_audio.py
输出:
    tests/fixtures/hello_singapore_zh.wav   (16kHz mono float->int16)
    tests/fixtures/hello_singapore_zh.txt   (参考文本)
"""

import os
import sys

# 允许从项目根导入
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import soundfile as sf
import torch
import torchaudio

from models.tts import TTSEngine
from config import SAMPLE_RATE


TEXT = "你好，我正在新加坡与你通话。"
LANGUAGE = "Chinese"

FIX_DIR = os.path.join(ROOT, "tests", "fixtures")
os.makedirs(FIX_DIR, exist_ok=True)
WAV_PATH = os.path.join(FIX_DIR, "hello_singapore_zh.wav")
TXT_PATH = os.path.join(FIX_DIR, "hello_singapore_zh.txt")


def main():
    print(f"[gen] 加载 TTS 模型…")
    tts = TTSEngine()
    print(f"[gen] 合成文本: {TEXT!r}  语言={LANGUAGE}")
    audio, sr = tts.synthesize(TEXT, LANGUAGE)

    # audio 可能是 torch.Tensor 或 numpy array
    if isinstance(audio, torch.Tensor):
        wav = audio.detach().to(torch.float32).cpu()
    else:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # -> (1, T)

    print(f"[gen] TTS 输出: shape={tuple(wav.shape)} sr={sr}")

    # 重采样到管线统一采样率 16k，与 pipeline.py 的做法一致
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        print(f"[gen] 重采样 {sr} -> {SAMPLE_RATE}")

    # 归一化到 [-1, 1] 防止 clipping
    peak = wav.abs().max().item()
    if peak > 1.0:
        wav = wav / peak

    # 用 soundfile 写 PCM16 wav（torchaudio.save 在新版本下需要 torchcodec）
    wav_np = wav.squeeze(0).numpy().astype(np.float32)
    sf.write(WAV_PATH, wav_np, SAMPLE_RATE, subtype="PCM_16")
    with open(TXT_PATH, "w") as f:
        f.write(TEXT + "\n")

    dur = wav_np.shape[-1] / SAMPLE_RATE
    print(f"[gen] 已保存: {WAV_PATH}  时长={dur:.2f}s")
    print(f"[gen] 参考文本: {TXT_PATH}")


if __name__ == "__main__":
    main()
