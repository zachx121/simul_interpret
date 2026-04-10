"""ASR 模块独立测试

对每个 case:
  1. 加载 wav 和参考文本
  2. 按 FRAME_SIZE=512 逐帧喂入 StreamingASR.feed_audio
  3. 记录所有 feed_audio 的非 None 返回（标点触发分支）
  4. finish_session 拿尾部（兜底分支）
  5. 拼接 → 与参考文本做 CER 断言

退出码:
  0 = all PASS
  1 = any case FAIL（带诊断输出）
"""

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


FIX_DIR = os.path.join(THIS_DIR, "fixtures")


CASES = [
    {
        "name": "hello_singapore_zh (~3s)",
        "wav": "hello_singapore_zh.wav",
        "txt": "hello_singapore_zh.txt",
        "cer_threshold": 0.15,
        # macOS Transformers 后端：feed_audio 只缓冲不 emit，全靠 finish_session
        "expect_feed_emissions_ge": 0,
        "require_nonempty_output": True,
    },
    {
        "name": "two_sentences_zh (~5s)",
        "wav": "two_sentences_zh.wav",
        "txt": "two_sentences_zh.txt",
        "cer_threshold": 0.15,
        "expect_feed_emissions_ge": 0,
        "require_nonempty_output": True,
    },
]


def run_case(asr: StreamingASR, case: dict) -> list[str]:
    """返回问题列表；为空表示通过。"""
    name = case["name"]
    wav_path = os.path.join(FIX_DIR, case["wav"])
    txt_path = os.path.join(FIX_DIR, case["txt"])

    audio, sr = load_wav(wav_path)
    ref = open(txt_path).read().strip()
    dur = len(audio) / sr

    print(f"\n[case] {name}")
    print(f"    wav      : {wav_path}")
    print(f"    sr       : {sr}  samples={len(audio)}  dur={dur:.2f}s")
    print(f"    ref      : {ref!r}")

    if sr != SAMPLE_RATE:
        return [f"sample rate mismatch: expected {SAMPLE_RATE}, got {sr}"]

    tlog = TimestampLog()
    asr.start_session()
    tlog.mark("start_session")

    feed_emissions: list[str] = []
    feed_calls = 0
    t_feed_total = 0.0

    for i, frame in enumerate(iter_frames(audio, FRAME_SIZE)):
        t0 = time.time()
        result = asr.feed_audio(frame)
        t_feed_total += time.time() - t0
        feed_calls += 1
        if result:
            feed_emissions.append(result)
            tlog.mark(f"emit@frame{i} (t_audio={i*FRAME_SIZE/SAMPLE_RATE:.2f}s): {result!r}")

    tlog.mark(f"all frames fed ({feed_calls} frames, feed_audio total {t_feed_total:.2f}s)")

    t0 = time.time()
    tail = asr.finish_session()
    tlog.mark(f"finish_session ({time.time()-t0:.2f}s): {tail!r}")

    hyp_parts = list(feed_emissions)
    if tail:
        hyp_parts.append(tail)
    hyp = "".join(hyp_parts)

    err = cer(ref, hyp)

    print(f"    hyp      : {hyp!r}")
    print(f"    CER      : {err:.3f}  (threshold < {case['cer_threshold']})")
    print(f"    emissions: feed_audio={len(feed_emissions)}  tail={'yes' if tail else 'no'}  "
          f"(expect feed>={case['expect_feed_emissions_ge']})")
    print(f"    timeline :")
    print(tlog.dump())

    problems = []
    if case["require_nonempty_output"] and not hyp:
        problems.append("hypothesis is empty (both feed_audio and finish_session returned nothing)")
    if err >= case["cer_threshold"]:
        problems.append(f"CER {err:.3f} >= threshold {case['cer_threshold']}")
    if len(feed_emissions) < case["expect_feed_emissions_ge"]:
        problems.append(
            f"feed_audio emissions={len(feed_emissions)}, "
            f"expected >= {case['expect_feed_emissions_ge']}"
        )
    return problems


def main() -> None:
    print(f"[test_asr] DEVICE={DEVICE}  _USE_VLLM={_USE_VLLM}  "
          f"ASR_CHUNK_SIZE_SEC={ASR_CHUNK_SIZE_SEC}  FRAME_SIZE={FRAME_SIZE}")
    print(f"[test_asr] loading StreamingASR ...")
    t0 = time.time()
    asr = StreamingASR()
    print(f"[test_asr] loaded in {time.time()-t0:.1f}s")

    all_problems: list[tuple[str, list[str]]] = []
    for case in CASES:
        try:
            problems = run_case(asr, case)
        except Exception as e:
            traceback.print_exc()
            problems = [f"exception: {type(e).__name__}: {e}"]
        if problems:
            all_problems.append((case["name"], problems))

    print("\n" + "=" * 68)
    if all_problems:
        print("[test_asr] RESULT: FAIL")
        for name, problems in all_problems:
            print(f"  {name}:")
            for p in problems:
                print(f"    - {p}")
        sys.exit(1)
    print("[test_asr] RESULT: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
