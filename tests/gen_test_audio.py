"""生成固定文本的测试音频，用于后续 ASR/流式管线测试。

用法:
    python tests/gen_test_audio.py
输出（已存在则跳过，避免重复加载 TTS）:
    tests/fixtures/hello_singapore_zh.wav    (短句, ~3s, 兜底路径)
    tests/fixtures/hello_singapore_zh.txt
    tests/fixtures/two_sentences_zh.wav      (两句, ~6-7s, 标点触发路径)
    tests/fixtures/two_sentences_zh.txt
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


FIX_DIR = os.path.join(ROOT, "tests", "fixtures")
os.makedirs(FIX_DIR, exist_ok=True)


FIXTURES = [
    {
        "name": "hello_singapore_zh",
        "text": "你好，我正在新加坡与你通话。",
        "language": "Chinese",
    },
    {
        "name": "two_sentences_zh",
        "text": "你好，我正在新加坡与你通话。今天新加坡的天气非常好。",
        "language": "Chinese",
    },
]


def synth_and_save(tts: TTSEngine, fix: dict):
    name = fix["name"]
    text = fix["text"]
    language = fix["language"]
    wav_path = os.path.join(FIX_DIR, f"{name}.wav")
    txt_path = os.path.join(FIX_DIR, f"{name}.txt")

    if os.path.exists(wav_path) and os.path.exists(txt_path):
        print(f"[gen] skip {name}: 已存在 -> {wav_path}")
        return

    print(f"[gen] 合成 {name}: {text!r}  语言={language}")
    audio, sr = tts.synthesize(text, language)

    if isinstance(audio, torch.Tensor):
        wav = audio.detach().to(torch.float32).cpu()
    else:
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)  # -> (1, T)

    print(f"[gen]   TTS 输出: shape={tuple(wav.shape)} sr={sr}")

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        print(f"[gen]   重采样 {sr} -> {SAMPLE_RATE}")

    peak = wav.abs().max().item()
    if peak > 1.0:
        wav = wav / peak

    wav_np = wav.squeeze(0).numpy().astype(np.float32)
    sf.write(wav_path, wav_np, SAMPLE_RATE, subtype="PCM_16")
    with open(txt_path, "w") as f:
        f.write(text + "\n")

    dur = wav_np.shape[-1] / SAMPLE_RATE
    print(f"[gen]   已保存: {wav_path}  时长={dur:.2f}s")


def main():
    # 先看看是否有 fixture 需要生成，全部已存在的话可以跳过 TTS 加载
    pending = [
        f for f in FIXTURES
        if not (
            os.path.exists(os.path.join(FIX_DIR, f["name"] + ".wav"))
            and os.path.exists(os.path.join(FIX_DIR, f["name"] + ".txt"))
        )
    ]
    if not pending:
        print("[gen] 所有 fixture 已存在，无需生成。")
        return

    print(f"[gen] 加载 TTS 模型… (待生成 {len(pending)} 个)")
    tts = TTSEngine()
    for fix in FIXTURES:
        synth_and_save(tts, fix)


if __name__ == "__main__":
    main()
