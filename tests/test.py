
import os
import sys
import time
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, ROOT)
sys.path.insert(0, THIS_DIR)

from _utils import FRAME_SIZE, TimestampLog, cer, iter_frames, load_wav  # noqa: E402
from config import ASR_CHUNK_SIZE_SEC, DEVICE, SAMPLE_RATE  # noqa: E402
from models.asr import _USE_VLLM, StreamingASR  # noqa: E402


def test_asr():
    # 调用 server 里的 vad 扫一遍 wav 决定哪里插入 finish_session，模拟流式地输出 ASR 文本结果
    import torch
    from config import SILENCE_TIMEOUT_MS, VAD_THRESHOLD
    from models.vad import VoiceDetector

    # ffmpeg -i data/航班.wav -ar 16000 data/航班_16k.wav
    # ffmpeg -i data/航班_head10s.wav -ar 16000 data/航班_16k_head10s.wav
    audio_fp = "/Users/bytedance/0-Code/YiSounda/data/航班_16k.wav"
    audio_fp = "/Users/bytedance/0-Code/YiSounda/data/航班_16k_head10s.wav"
    print(f"[test_asr] DEVICE={DEVICE}  _USE_VLLM={_USE_VLLM}  "
          f"ASR_CHUNK_SIZE_SEC={ASR_CHUNK_SIZE_SEC}  FRAME_SIZE={FRAME_SIZE}")
    print(f"[test_asr] audio_fp = {audio_fp}")
    print(f"[test_asr] loading StreamingASR ...")
    t0 = time.time()
    asr = StreamingASR()
    print(f"[test_asr] loaded in {time.time()-t0:.1f}s")

    vad = VoiceDetector(threshold=VAD_THRESHOLD)
    SILENCE_TIMEOUT_MS = 250  #
    silence_frames = int(SILENCE_TIMEOUT_MS / (1000 / (SAMPLE_RATE / FRAME_SIZE)))

    audio, sr = load_wav(audio_fp)
    assert sr == SAMPLE_RATE, f"sr={sr}, expected {SAMPLE_RATE}"

    tlog = TimestampLog()
    silence_count = 0
    is_speaking = False
    asr_active = False

    for i, frame in enumerate(iter_frames(audio, FRAME_SIZE)):
        t_audio = i * FRAME_SIZE / SAMPLE_RATE

        if vad.is_speech(torch.tensor(frame)):
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                asr.start_session()
                asr_active = True
                tlog.mark(f"[t={t_audio:6.2f}s] 🎤 speech onset → start_session")
            sentence = asr.feed_audio(frame)
            if sentence:
                tlog.mark(f"[t={t_audio:6.2f}s] 📝 emit: {sentence!r}")
        elif is_speaking:
            silence_count += 1
            if asr_active:
                sentence = asr.feed_audio(frame)
                if sentence:
                    tlog.mark(f"[t={t_audio:6.2f}s] 📝 emit (silence): {sentence!r}")
            if silence_count >= silence_frames:
                if asr_active:
                    remaining = asr.finish_session()
                    tlog.mark(f"[t={t_audio:6.2f}s] 🔇 silence flush: {remaining!r}")
                    asr_active = False
                silence_count = 0
                is_speaking = False
                vad.reset()

    if asr_active:
        remaining = asr.finish_session()
        tlog.mark(f"[t={len(audio)/sr:6.2f}s] ⏹  eof flush: {remaining!r}")

    print(tlog.dump())


def test_asr_translate():
    # 调用 server 里的 vad 扫一遍 wav 决定哪里插入 finish_session，模拟流式地输出 ASR 文本结果
    import torch
    from config import SILENCE_TIMEOUT_MS, VAD_THRESHOLD
    from models.vad import VoiceDetector

    # ffmpeg -i data/航班.wav -ar 16000 data/航班_16k.wav
    # ffmpeg -i data/航班_head10s.wav -ar 16000 data/航班_16k_head10s.wav
    audio_fp = "/Users/bytedance/0-Code/YiSounda/data/航班_16k.wav"
    audio_fp = "/Users/bytedance/0-Code/YiSounda/data/航班_16k_head10s.wav"
    print(f"[test_asr] DEVICE={DEVICE}  _USE_VLLM={_USE_VLLM}  "
          f"ASR_CHUNK_SIZE_SEC={ASR_CHUNK_SIZE_SEC}  FRAME_SIZE={FRAME_SIZE}")
    print(f"[test_asr] audio_fp = {audio_fp}")
    print(f"[test_asr] loading StreamingASR ...")
    t0 = time.time()
    asr = StreamingASR()
    print(f"[test_asr] loaded in {time.time()-t0:.1f}s")

    vad = VoiceDetector(threshold=VAD_THRESHOLD)
    SILENCE_TIMEOUT_MS = 250  #
    silence_frames = int(SILENCE_TIMEOUT_MS / (1000 / (SAMPLE_RATE / FRAME_SIZE)))

    audio, sr = load_wav(audio_fp)
    assert sr == SAMPLE_RATE, f"sr={sr}, expected {SAMPLE_RATE}"

    tlog = TimestampLog()
    silence_count = 0
    is_speaking = False
    asr_active = False

    for i, frame in enumerate(iter_frames(audio, FRAME_SIZE)):
        t_audio = i * FRAME_SIZE / SAMPLE_RATE

        if vad.is_speech(torch.tensor(frame)):
            silence_count = 0
            if not is_speaking:
                is_speaking = True
                asr.start_session()
                asr_active = True
                tlog.mark(f"[t={t_audio:6.2f}s] 🎤 speech onset → start_session")
            sentence = asr.feed_audio(frame)
            if sentence:
                tlog.mark(f"[t={t_audio:6.2f}s] 📝 emit: {sentence!r}")
        elif is_speaking:
            silence_count += 1
            if asr_active:
                sentence = asr.feed_audio(frame)
                if sentence:
                    tlog.mark(f"[t={t_audio:6.2f}s] 📝 emit (silence): {sentence!r}")
            if silence_count >= silence_frames:
                if asr_active:
                    remaining = asr.finish_session()
                    tlog.mark(f"[t={t_audio:6.2f}s] 🔇 silence flush: {remaining!r}")
                    asr_active = False
                silence_count = 0
                is_speaking = False
                vad.reset()

    if asr_active:
        remaining = asr.finish_session()
        tlog.mark(f"[t={len(audio)/sr:6.2f}s] ⏹  eof flush: {remaining!r}")

    print(tlog.dump())


if __name__ == '__main__':
    test_asr()
