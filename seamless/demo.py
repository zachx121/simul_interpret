"""
python3 -c "from seamless_communication.streaming.agents.seamless_streaming_s2t import SeamlessStreamingS2TAgent;print('ok')"


SeamlessStreaming 最小流式 Demo.

- 写死一个 wav 路径，`stream_wav_chunks` 按 320ms 块产出音频，支持 `cap` 参数。
- `mode=s2tt` 流式打印翻译文本；`mode=s2st` 流式收集翻译音频并存到 out.wav。
- 先调通再优化，边界条件和性能都没管。

参数
  --no-early-stop
  --max-len-a
  --max-len-b
  --min-unit-chunk-size 25 卡顿跳跃感太明显了还是用默认的50吧
  --first-unit-chunk-size 40 首包用小阈值还行
  --decision-threshold 阈值越低，越容易更早出第一段翻译文本，但是更容易出现不稳定 partial、重复、后续改写
运行示例：
  python3 demo.py --mode s2tt --tgt-lang eng --audio outputs_tts_16khz.wav
  python3 demo.py --mode s2tt --tgt-lang eng --audio 米奇沃克斯.wav
    ```
    [t= 0.81s chunk=320ms   6/55 rms=0.1528 peak=0.4420] 📝 Hi, I
    [t= 1.19s chunk=320ms   9/55 rms=0.0000 peak=0.0000] 📝 'm in Singapore to talk
    [t= 1.30s chunk=320ms  10/55 rms=0.0819 peak=0.2890] 📝 to you
    [t= 1.96s chunk=320ms  17/55 rms=0.0793 peak=0.3585] 📝 . The weather in Singapore is very
    [t= 2.06s chunk=320ms  18/55 rms=0.1408 peak=0.4272] 📝 nice
    [t= 2.37s chunk=320ms  22/55 rms=0.1157 peak=0.4364] 📝 today
    [t= 2.47s chunk=320ms  23/55 rms=0.1035 peak=0.5221] 📝 .
    [t= 2.85s chunk=320ms  33/55 rms=0.1176 peak=0.4702] 📝 Let's talk
    [t= 3.00s chunk=320ms  35/55 rms=0.0940 peak=0.2858] 📝 about
    [t= 3.72s chunk=320ms  41/55 rms=0.1386 peak=0.5683] 📝 how to accurately calculate the final result of a flow meter
    [t= 3.94s chunk=320ms  44/55 rms=0.0919 peak=0.3621] 📝 and
    [t= 4.10s chunk=320ms  45/55 rms=0.1091 peak=0.4345] 📝 reasonably estimate
    [t= 4.41s chunk=320ms  49/55 rms=0.1236 peak=0.5049] 📝 how
    [t= 4.78s chunk=320ms  53/55 rms=0.0000 peak=0.0001] 📝 much error
    [t= 5.12s chunk=320ms  55/59] 📝 it will cause to the entire system.
    [demo] total elapsed: 5.4s
    ```
  python3 demo.py --mode s2st --tgt-lang eng --audio outputs_tts_16khz.wav --out out.wav
  python3 demo.py --mode s2st --tgt-lang eng --audio outputs_tts_16khz.wav --out out.wav --first-unit-chunk-size 40
  python3 demo.py --mode s2st --tgt-lang eng --audio 米奇沃克斯.wav --out out.wav --first-unit-chunk-size 40

    ```
    [demo] built in 14.8s
    [t= 1.16s chunk=320ms   6/55 rms=0.1528 peak=0.4420] 🔊 +15360 samples (~0.96s)
    [t= 1.82s chunk=320ms   9/55 rms=0.0000 peak=0.0000] 🔊 +20160 samples (~1.26s)
    [t= 3.25s chunk=320ms  17/55 rms=0.0793 peak=0.3585] 🔊 +35520 samples (~2.22s)
    [t= 4.44s chunk=320ms  23/55 rms=0.1035 peak=0.5221] 🔊 +13440 samples (~0.84s)
    [t= 5.25s chunk=320ms  35/55 rms=0.0940 peak=0.2858] 🔊 +13440 samples (~0.84s)
    [t= 6.44s chunk=320ms  41/55 rms=0.1386 peak=0.5683] 🔊 +45440 samples (~2.84s)
    [t= 7.18s chunk=320ms  45/55 rms=0.1091 peak=0.4345] 🔊 +20480 samples (~1.28s)
    [t= 8.96s chunk=320ms  55/59] 🔊 +39040 samples (~2.44s)
    [demo] wrote out.wav: 12.68s
    [demo] total elapsed: 9.3s
    ```
  # --debug模式是重新执行了ASR、S2TT的，所以耗时会显著增加
  python3 demo.py --mode s2st --tgt-lang eng --audio outputs_tts_16khz.wav --out out.wav  --debug


"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from types import MethodType
from typing import Any, Iterator

import numpy as np
import soundfile as sf
import torch

from simuleval.data.segments import SpeechSegment

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SEAMLESS_SRC = PROJECT_ROOT / "seamless_communication" / "src"
if LOCAL_SEAMLESS_SRC.exists():
    sys.path.insert(0, str(LOCAL_SEAMLESS_SRC))

AUDIO_PATH = "/mlx_devbox/users/tong.zhou1/playground/seamless/outputs_tts_16khz.wav"
AUDIO_TEXT = "你好，我正在新加坡与你通话。今天新加坡的天气非常好，你那边呢？现在我们来讨论一下，怎么样精确地计算一个流式计数器的最终结果，并且合理地预估它会对整个系统带来多少误差。"
SAMPLE_RATE = 16000
DEFAULT_CHUNK_MS = 320  # SeamlessStreaming 默认 source_segment_size


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def stream_wav_chunks(
    path: str, cap_sec: float | None = None, chunk_ms: int = DEFAULT_CHUNK_MS
) -> Iterator[np.ndarray]:
    """读 wav，按 chunk_ms 切块流式产出 float32 数组；达到 cap_sec 后停。"""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    assert sr == SAMPLE_RATE, f"need {SAMPLE_RATE}Hz, got {sr}"
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if cap_sec is not None:
        audio = audio[: int(sr * cap_sec)]
    chunk_size = int(sr * chunk_ms / 1000)  # 16k/320ms = 5120 samples
    for start in range(0, len(audio), chunk_size):
        yield audio[start : start + chunk_size]


def get_total_chunks(
    path: str, cap_sec: float | None = None, chunk_ms: int = DEFAULT_CHUNK_MS
) -> int:
    """根据音频长度预先计算总 chunk 数。"""
    info = sf.info(path)
    assert info.samplerate == SAMPLE_RATE, f"need {SAMPLE_RATE}Hz, got {info.samplerate}"

    frames = info.frames
    if cap_sec is not None:
        frames = min(frames, int(info.samplerate * cap_sec))

    chunk_size = int(info.samplerate * chunk_ms / 1000)
    return (frames + chunk_size - 1) // chunk_size if frames > 0 else 0


def get_chunk_volume_stats(chunk: np.ndarray) -> tuple[float, float]:
    """返回当前输入 chunk 的 RMS 和 peak，便于判断是否近似静音。"""
    if chunk.size == 0:
        return 0.0, 0.0

    chunk32 = np.asarray(chunk, dtype=np.float32)
    rms = float(np.sqrt(np.mean(np.square(chunk32))))
    peak = float(np.max(np.abs(chunk32)))
    return rms, peak


def get_agent_diagnostics(agent: Any) -> str:
    """提取 VAD/decoder 的最小可观测状态，辅助判断是未切段还是队列堆积。"""
    vad_finished = None
    next_q = None
    input_q = None
    dec_src_len = None
    dec_tgt_len = None
    dec_tgt_finished = None

    for module in getattr(agent, "module_list", []):
        name = module.__class__.__name__
        states = getattr(module, "states", None)
        if states is None:
            continue

        if "SileroVAD" in name:
            vad_finished = getattr(states, "source_finished", None)
            input_queue = getattr(states, "input_queue", None)
            next_queue = getattr(states, "next_input_queue", None)
            if input_queue is not None and hasattr(input_queue, "qsize"):
                input_q = input_queue.qsize()
            if next_queue is not None and hasattr(next_queue, "qsize"):
                next_q = next_queue.qsize()

        if "DecoderAgent" in name and hasattr(states, "target_indices"):
            dec_src_len = getattr(states, "source_len", None)
            target_indices = getattr(states, "target_indices", None)
            if target_indices is not None:
                dec_tgt_len = len(target_indices)
            dec_tgt_finished = getattr(states, "target_finished", None)

    parts = []
    if vad_finished is not None:
        parts.append(f"vad_finished={int(bool(vad_finished))}")
    if input_q is not None:
        parts.append(f"vad_q={input_q}")
    if next_q is not None:
        parts.append(f"next_q={next_q}")
    if dec_src_len is not None:
        parts.append(f"dec_src={dec_src_len}")
    if dec_tgt_len is not None:
        parts.append(f"dec_tgt={dec_tgt_len}")
    if dec_tgt_finished is not None:
        parts.append(f"dec_finished={int(bool(dec_tgt_finished))}")

    return " ".join(parts)


def build_agent(
    mode: str,
    tgt_lang: str,
    device: str,
    decision_threshold: float = 0.5,
    max_len_b: int = 200,
    task: str | None = None,
    silence_limit_ms: int | None = None,
    min_unit_chunk_size: int = 50,
    first_unit_chunk_size: int | None = None,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    min_starting_wait_w2vbert: int = 192,
):
    """按 mode 选 S2TT 或 S2ST agent（带 VAD + Detokenizer 变体），构造 Namespace 并 from_args。

    为什么用 VAD 变体：
      `UnitYAgentPipeline.pop` 遇到 "源未结束但输出 finished=True" 会 reset 全部 state。
      non-VAD 变体在句末 reset 时下一句的音频已经铺开，encoder 没机会重建上下文 → 翻译沉默。
      VAD 变体用 SileroVAD 在静音处切段，reset 发生在干净时刻，多句连续输入才能正常处理。
    VAD 变体同时带 DetokenizerAgent，输出直接是英文字符串而非 SentencePiece token。
    """
    from seamless_communication.streaming.agents.seamless_streaming_s2st import (
        SeamlessStreamingS2STVADAgent,
    )
    from seamless_communication.streaming.agents.seamless_streaming_s2t import (
        SeamlessStreamingS2TVADAgent,
    )

    task = task or mode
    AgentCls = SeamlessStreamingS2STVADAgent if task == "s2st" else SeamlessStreamingS2TVADAgent

    # agent.add_args 只注册 agent 级的 flag（--task / --tgt-lang / --dtype / ...）
    # --device / --fp16 / --source-segment-size 是 SimulEval general_parser 的，
    # 这里走 argparse 拿不到，后面手工 setattr。
    parser = argparse.ArgumentParser()
    AgentCls.add_args(parser)

    argv = [
        "--task", task,
        "--tgt-lang", tgt_lang,
        "--dtype", "fp16" if device.startswith("cuda") else "fp32",
        "--min-starting-wait-w2vbert", str(min_starting_wait_w2vbert),
        "--decision-threshold", str(decision_threshold),
        # "--no-early-stop",
        "--max-len-a", "1",
        "--max-len-b", str(max_len_b),
    ]
    if silence_limit_ms is not None:
        argv += ["--silence-limit-ms", str(silence_limit_ms)]
    if task == "s2st":
        argv += ["--min-unit-chunk-size", str(min_unit_chunk_size)]
        if first_unit_chunk_size is not None:
            argv += ["--first-unit-chunk-size", str(first_unit_chunk_size)]

    args = parser.parse_args(argv)

    # 手工注入 SimulEval general_parser 的字段（load_model 会读）
    args.device = device
    args.fp16 = device.startswith("cuda")
    args.source_segment_size = chunk_ms

    return AgentCls.from_args(args)


def _emit(
    seg,
    i: int,
    total_chunks: int | None,
    t0: float,
    audio_buf: list[np.ndarray] | None,
    chunk_ms: int,
    chunk_rms: float | None = None,
    chunk_peak: float | None = None,
    always_log_prefix: bool = False,
    label: str | None = None,
) -> None:
    """把一个 segment 打印/收集起来。"""
    dt = time.time() - t0
    if total_chunks is None:
        prefix = f"[t={dt:5.2f}s]"
    else:
        volume = ""
        if chunk_rms is not None and chunk_peak is not None:
            volume = f" rms={chunk_rms:.4f} peak={chunk_peak:.4f}"
        prefix = (
            f"[t={dt:5.2f}s chunk={chunk_ms}ms {i:3d}/{total_chunks}{volume}]"
        )

    if label:
        prefix = f"[{label}] {prefix}"

    if seg is None or getattr(seg, "is_empty", False):
        if always_log_prefix:
            print(prefix)
        return

    if seg.data_type == "text" and seg.content:
        print(f"{prefix} 📝 {seg.content}")
    elif seg.data_type == "speech" and seg.content:
        arr = np.asarray(seg.content, dtype=np.float32)
        if audio_buf is None:
            raise ValueError("audio_buf is required when emitting speech segments")
        audio_buf.append(arr)
        seconds = len(arr) / SAMPLE_RATE
        print(f"{prefix} 🔊 +{len(arr)} samples (~{seconds:.2f}s)")
    elif always_log_prefix:
        print(prefix)


def _find_decoder_module(agent: Any) -> Any | None:
    for module in getattr(agent, "module_list", []):
        if "DecoderAgent" in module.__class__.__name__:
            return module
    return None


def enable_decoder_debug(agent: Any) -> None:
    decoder = _find_decoder_module(agent)
    if decoder is None or getattr(decoder, "_debug_enabled", False):
        return

    original_run_decoder = decoder.run_decoder

    def wrapped_run_decoder(self, states, pred_indices):
        index, prob, decoder_features = original_run_decoder(states, pred_indices)
        self._last_debug = {
            "index": int(index),
            "token": self.text_tokenizer.model.index_to_token(int(index)),
            "prob": float(prob),
            "pred_len": len(pred_indices),
            "target_len": len(states.target_indices),
            "source_finished": bool(states.source_finished),
        }
        return index, prob, decoder_features

    decoder.run_decoder = MethodType(wrapped_run_decoder, decoder)
    decoder._debug_enabled = True


def get_decoder_debug_line(agent: Any, prev_target_len: int) -> tuple[str | None, int]:
    decoder = _find_decoder_module(agent)
    if decoder is None:
        return None, prev_target_len

    states = getattr(decoder, "states", None)
    if states is None or not hasattr(states, "target_indices"):
        return None, prev_target_len

    target_len = len(states.target_indices)
    delta = target_len - prev_target_len
    last = getattr(decoder, "_last_debug", None)
    detok_buf = None
    for module in getattr(agent, "module_list", []):
        if "Detokenizer" in module.__class__.__name__:
            detok_states = getattr(module, "states", None)
            if detok_states is not None and hasattr(detok_states, "source"):
                detok_buf = len(detok_states.source)
            break

    parts = [f"tgt={target_len}", f"Δ={delta}"]
    if detok_buf is not None:
        parts.append(f"detok_buf={detok_buf}")
    if last is not None:
        parts.append(f"last={last['token']}")
        parts.append(f"p={last['prob']:.3f}")
        parts.append(f"pred_len={last['pred_len']}")
        parts.append(f"src_finished={int(last['source_finished'])}")
    return " ".join(parts), target_len


def run(
    mode: str,
    tgt_lang: str = "eng",
    cap_sec: float | None = None,
    audio_path: str = AUDIO_PATH,
    out_wav: str = "out.wav",
    decision_threshold: float = 0.5,
    max_len_b: int = 200,
    src_lang: str = "cmn",
    log_asr: bool = False,
    tail_silence_ms: int = 1000,
    log_decoder: bool = False,
    silence_limit_ms: int | None = None,
    min_unit_chunk_size: int = 50,
    debug: bool = False,
    chunk_ms: int = DEFAULT_CHUNK_MS,
    first_unit_chunk_size: int | None = None,
    min_starting_wait_w2vbert: int = 192,
) -> None:
    device = pick_device()
    print(f"[demo] device={device}  mode={mode}  tgt={tgt_lang}  cap={cap_sec}s")
    print(f"[demo] audio={audio_path}")

    print(f"[demo] building agent ...")
    t = time.time()
    agent = build_agent(
        mode,
        tgt_lang,
        device,
        decision_threshold,
        max_len_b,
        silence_limit_ms=silence_limit_ms,
        min_unit_chunk_size=min_unit_chunk_size,
        first_unit_chunk_size=first_unit_chunk_size,
        chunk_ms=chunk_ms,
        min_starting_wait_w2vbert=min_starting_wait_w2vbert,
    )
    aux_agents: list[tuple[str, Any, str]] = []
    if log_asr or debug:
        aux_agents.append((
            "ASR",
            build_agent(
            "s2tt",
            src_lang,
            device,
            decision_threshold,
            max_len_b,
            task="asr",
            silence_limit_ms=silence_limit_ms,
            min_unit_chunk_size=min_unit_chunk_size,
            first_unit_chunk_size=first_unit_chunk_size,
            chunk_ms=chunk_ms,
            min_starting_wait_w2vbert=min_starting_wait_w2vbert,
            ),
            src_lang,
        ))
    if debug and mode == "s2st":
        aux_agents.append((
            "S2TT",
            build_agent(
                "s2tt",
                tgt_lang,
                device,
                decision_threshold,
                max_len_b,
                task="s2tt",
                silence_limit_ms=silence_limit_ms,
                min_unit_chunk_size=min_unit_chunk_size,
                first_unit_chunk_size=first_unit_chunk_size,
                chunk_ms=chunk_ms,
                min_starting_wait_w2vbert=min_starting_wait_w2vbert,
            ),
            tgt_lang,
        ))
    if log_decoder:
        enable_decoder_debug(agent)
    print(f"[demo] built in {time.time()-t:.1f}s")

    audio_buf: list[np.ndarray] = []
    total_chunks = get_total_chunks(audio_path, cap_sec=cap_sec, chunk_ms=chunk_ms)
    tail_chunk_count = max(0, (tail_silence_ms + chunk_ms - 1) // chunk_ms)
    prev_decoder_target_len = 0
    # 以开始流式推理的时间为 t0，排除上面模型加载时间。
    t0 = time.time()

    for i, chunk in enumerate(stream_wav_chunks(audio_path, cap_sec=cap_sec, chunk_ms=chunk_ms)):
        chunk_rms, chunk_peak = get_chunk_volume_stats(chunk)
        seg = SpeechSegment(
            content=chunk,
            sample_rate=SAMPLE_RATE,
            tgt_lang=tgt_lang,
            finished=False,
        )
        out = agent.pushpop(seg)
        _emit(
            out,
            i,
            total_chunks,
            t0,
            audio_buf,
            chunk_ms,
            chunk_rms=chunk_rms,
            chunk_peak=chunk_peak,
            always_log_prefix=debug,
        )
        for label, aux_agent, aux_tgt_lang in aux_agents:
            aux_out = aux_agent.pushpop(
                SpeechSegment(
                    content=chunk,
                    sample_rate=SAMPLE_RATE,
                    tgt_lang=aux_tgt_lang,
                    finished=False,
                )
            )
            _emit(
                aux_out,
                i,
                total_chunks,
                t0,
                None,
                chunk_ms,
                chunk_rms=chunk_rms,
                chunk_peak=chunk_peak,
                always_log_prefix=debug,
                label=label,
            )
        if log_decoder:
            decoder_line, prev_decoder_target_len = get_decoder_debug_line(
                agent, prev_decoder_target_len
            )
            if decoder_line is not None:
                print(f"[DEC] [chunk={i:3d}/{total_chunks}] {decoder_line}")
        diag = get_agent_diagnostics(agent)
        if diag:
            # print(f"[diag chunk={chunk_ms}ms {i:3d}/{total_chunks}] {diag}")
            pass

    if tail_chunk_count > 0:
        silence = np.zeros(int(SAMPLE_RATE * chunk_ms / 1000), dtype=np.float32)
        extended_total_chunks = total_chunks + tail_chunk_count
        for tail_i in range(tail_chunk_count):
            i = total_chunks + tail_i
            out = agent.pushpop(
                SpeechSegment(
                    content=silence,
                    sample_rate=SAMPLE_RATE,
                    tgt_lang=tgt_lang,
                    finished=False,
                )
            )
            _emit(out, i, extended_total_chunks, t0, audio_buf, chunk_ms)
            for label, aux_agent, aux_tgt_lang in aux_agents:
                aux_out = aux_agent.pushpop(
                    SpeechSegment(
                        content=silence,
                        sample_rate=SAMPLE_RATE,
                        tgt_lang=aux_tgt_lang,
                        finished=False,
                    )
                )
                _emit(
                    aux_out,
                    i,
                    extended_total_chunks,
                    t0,
                    None,
                    chunk_ms,
                    always_log_prefix=debug,
                    label=label,
                )
            if log_decoder:
                decoder_line, prev_decoder_target_len = get_decoder_debug_line(
                    agent, prev_decoder_target_len
                )
                if decoder_line is not None:
                    print(f"[DEC] [chunk={i:3d}/{extended_total_chunks}] {decoder_line}")

    # EOF flush：告诉 pipeline 输入结束，把剩余输出吐干净
    eof_seg = SpeechSegment(
        content=np.array([], dtype=np.float32),
        sample_rate=SAMPLE_RATE,
        tgt_lang=tgt_lang,
        finished=True,
    )
    out = agent.pushpop(eof_seg)
    _emit(out, -1, None, t0, audio_buf, chunk_ms)
    if log_decoder:
        decoder_line, prev_decoder_target_len = get_decoder_debug_line(
            agent, prev_decoder_target_len
        )
        if decoder_line is not None:
            print(f"[DEC] [eof] {decoder_line}")
    for label, aux_agent, aux_tgt_lang in aux_agents:
        aux_out = aux_agent.pushpop(
            SpeechSegment(
                content=np.array([], dtype=np.float32),
                sample_rate=SAMPLE_RATE,
                tgt_lang=aux_tgt_lang,
                finished=True,
            )
        )
        _emit(aux_out, -1, None, t0, None, chunk_ms, label=label)
    # drain 剩余输出，最多循环 500 次防死锁
    for _ in range(500):
        tail = agent.pop()
        _emit(tail, -1, None, t0, audio_buf, chunk_ms)
        if log_decoder:
            decoder_line, prev_decoder_target_len = get_decoder_debug_line(
                agent, prev_decoder_target_len
            )
            if decoder_line is not None:
                print(f"[DEC] [pop] {decoder_line}")
        aux_done = True
        for label, aux_agent, _ in aux_agents:
            aux_tail = aux_agent.pop()
            _emit(aux_tail, -1, None, t0, None, chunk_ms, label=label)
            aux_done = aux_done and (
                aux_tail is None or getattr(aux_tail, "finished", False)
            )
        if (tail is None or getattr(tail, "finished", False)) and aux_done:
            break

    if mode == "s2st" and audio_buf:
        wav = np.concatenate(audio_buf)
        sf.write(out_wav, wav, SAMPLE_RATE)
        print(f"[demo] wrote {out_wav}: {len(wav)/SAMPLE_RATE:.2f}s")

    print(f"[demo] total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["s2tt", "s2st"], default="s2tt")
    p.add_argument("--tgt-lang", default="eng")
    p.add_argument("--src-lang", default="cmn", help="源语言，用于 `--log-asr` 诊断")
    p.add_argument("--cap", type=float, default=None, help="只读前 N 秒音频；不传则处理完整音频")
    p.add_argument("--audio", default=AUDIO_PATH)
    p.add_argument("--out", default="out.wav")
    p.add_argument("--decision-threshold", type=float, default=0.5,
                   help="EMMA WRITE 阈值，调小 → 延迟更低但 BLEU 略降")
    p.add_argument("--max-len-b", type=int, default=200)
    p.add_argument("--log-asr", action="store_true", help="并行运行 ASR，打印源语言转写诊断")
    p.add_argument("--tail-silence-ms", type=int, default=1000,
                   help="文件输入结束后补零音频，帮助 VAD 切段并吐出尾部翻译")
    p.add_argument("--log-decoder", action="store_true",
                   help="打印翻译 decoder 的中间态，辅助判断为何停止出词")
    p.add_argument("--silence-limit-ms", type=int, default=None,
                   help="覆盖 Silero VAD 的切段静音阈值（毫秒）")
    p.add_argument("--min-unit-chunk-size", type=int, default=50,
                   help="S2ST 模式下攒多少个 unit 后再送 vocoder 合成")
    p.add_argument("--first-unit-chunk-size", type=int, default=None,
                   help="S2ST 模式下首个语音 chunk 的最小 unit 数；不传则与 --min-unit-chunk-size 相同")
    p.add_argument("--min-starting-wait-w2vbert", type=int, default=192,
                   help="encoder 起步前最少等待的声学帧数，默认 192")
    p.add_argument("--chunk-ms", type=int, default=DEFAULT_CHUNK_MS,
                   help="输入音频切块大小（毫秒），默认 320")
    p.add_argument("--debug", action="store_true",
                   help="打印逐 chunk 详情；在 s2st 下额外打印 ASR 和翻译文本")
    a = p.parse_args()
    run(a.mode, a.tgt_lang, a.cap, a.audio, a.out,
        a.decision_threshold, a.max_len_b, a.src_lang, a.log_asr,
        a.tail_silence_ms, a.log_decoder, a.silence_limit_ms,
        a.min_unit_chunk_size, a.debug, a.chunk_ms, a.first_unit_chunk_size,
        a.min_starting_wait_w2vbert)
