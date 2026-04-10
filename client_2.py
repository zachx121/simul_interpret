"""
文件输入的语音翻译客户端 — 读取本地音频文件，流式发送到服务端，接收翻译音频并保存。

使用方式：
  python client_2.py --input audio.mp3 --inp-lang zh --opt-lang en
  python client_2.py --input audio.mp3 --inp-lang zh --opt-lang en --output out.wav
"""

import asyncio
import argparse
import struct
import wave

import numpy as np
import soundfile as sf
import websockets

# ── 音频���数（需与服务端一致） ─────────────────
SEND_RATE = 16000
FRAME_SIZE = 512  # 需与服务端 config.FRAME_SIZE 一致
CHANNELS = 1


async def run(
    server_url: str,
    input_file: str,
    inp_lang: str,
    opt_lang: str,
    output_file: str | None = None,
):
    # 读取音频文件并重采样到 16kHz mono
    audio, sr = sf.read(input_file, dtype="float32", always_2d=True)
    # 转 mono
    if audio.shape[1] > 1:
        audio = audio.mean(axis=1)
    else:
        audio = audio[:, 0]

    # 重采样到 16kHz
    if sr != SEND_RATE:
        import torchaudio.functional as F
        import torch
        audio = F.resample(
            torch.from_numpy(audio), orig_freq=sr, new_freq=SEND_RATE
        ).numpy()

    # 转 PCM 16-bit bytes
    pcm_int16 = (audio * 32768).astype(np.int16)

    url = f"{server_url}/stream?inp_lang={inp_lang}&opt_lang={opt_lang}"
    print(f"连接服务: {url}")
    print(f"输入文件: {input_file} ({len(audio) / SEND_RATE:.1f}s)")

    received_chunks: list[bytes] = []

    async with websockets.connect(url) as ws:
        print("已连接，开始发送���频...")

        send_done = asyncio.Event()

        async def send_audio():
            """按 FRAME_SIZE 切分，模拟实时发送。"""
            total_frames = len(pcm_int16) // FRAME_SIZE
            for i in range(total_frames):
                start = i * FRAME_SIZE
                end = start + FRAME_SIZE
                chunk = pcm_int16[start:end].tobytes()
                await ws.send(chunk)
                # 模拟实时速率
                await asyncio.sleep(FRAME_SIZE / SEND_RATE)

            # ���送剩余不足一帧的部��（补零）
            remainder = len(pcm_int16) % FRAME_SIZE
            if remainder > 0:
                start = total_frames * FRAME_SIZE
                last = pcm_int16[start:]
                padded = np.zeros(FRAME_SIZE, dtype=np.int16)
                padded[:remainder] = last
                await ws.send(padded.tobytes())
                await asyncio.sleep(FRAME_SIZE / SEND_RATE)

            # 发送一段静音让服务端触发尾部提交
            silence = np.zeros(FRAME_SIZE, dtype=np.int16).tobytes()
            for _ in range(30):  # ~1s 静音
                await ws.send(silence)
                await asyncio.sleep(FRAME_SIZE / SEND_RATE)

            print("音频发送完毕，等待翻译结果...")
            send_done.set()

        async def receive_audio():
            """接收翻译后的音频。"""
            try:
                async for message in ws:
                    if isinstance(message, bytes):
                        received_chunks.append(message)
                        total_bytes = sum(len(c) for c in received_chunks)
                        duration = total_bytes / (SEND_RATE * 2)
                        print(f"\r  收到音频: {duration:.1f}s", end="", flush=True)
            except websockets.exceptions.ConnectionClosed:
                pass

        async def watchdog():
            """发送完后等待一段时间再关闭连接。"""
            await send_done.wait()
            await asyncio.sleep(10)  # 等服务端处理完
            await ws.close()

        await asyncio.gather(send_audio(), receive_audio(), watchdog())

    print()

    if not received_chunks:
        print("未收到翻译音频。")
        return

    # 合并所有音频
    all_pcm = b"".join(received_chunks)
    total_duration = len(all_pcm) / (SEND_RATE * 2)
    print(f"共收到 {total_duration:.1f}s 翻译音频")

    # 保存输出
    if output_file is None:
        import os
        base = os.path.splitext(input_file)[0]
        output_file = f"{base}_translated.wav"

    with wave.open(output_file, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SEND_RATE)
        wf.writeframes(all_pcm)

    print(f"已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="文件输入的语音翻译客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python client_2.py --input audio.mp3 --inp-lang zh --opt-lang en
  python client_2.py --input audio.mp3 --inp-lang zh --opt-lang en --output out.wav
""",
    )
    parser.add_argument("--server", default="ws://localhost:8765", help="服务地址")
    parser.add_argument("--input", required=True, help="输入音频文件路径 (mp3/wav/flac 等)")
    parser.add_argument("--inp-lang", required=True, help="输入语言 (zh/en/ja/ko/fr/de/es)")
    parser.add_argument("--opt-lang", required=True, help="输出语言 (zh/en/ja/ko/fr/de/es)")
    parser.add_argument("--output", default=None, help="输出 WAV 文件路径 (默认: <input>_translated.wav)")
    args = parser.parse_args()

    asyncio.run(run(args.server, args.input, args.inp_lang, args.opt_lang, args.output))


if __name__ == "__main__":
    main()
