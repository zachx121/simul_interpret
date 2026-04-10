"""
实时语音翻译客户端 — 麦克风采集 + 耳机播放。

双协程架构：
  send_audio   持续从麦克风读取 30ms 帧 → WebSocket 发送
  receive_audio 持续从 WebSocket 接收 → 耳机播放

使用方式：
  # 列出音频设备（找耳机编号）
  python client.py --list-devices

  # 中文 → 英文，耳机设备 3
  python client.py --inp-lang zh --opt-lang en --output-device 3
"""

import asyncio
import argparse

import pyaudio
import websockets

# ── 音频参数（需与服务端一致） ─────────────────
SEND_RATE = 16000
FRAME_MS = 32
FRAME_SIZE = 512  # 需与服务端 config.FRAME_SIZE 一致
CHANNELS = 1
FORMAT = pyaudio.paInt16


async def run(
    server_url: str,
    inp_lang: str,
    opt_lang: str,
    output_device: int | None = None,
):
    pa = pyaudio.PyAudio()

    # 麦克风
    mic = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_RATE,
        input=True,
        frames_per_buffer=FRAME_SIZE,
    )

    # 耳机
    spk_kwargs = dict(
        format=FORMAT,
        channels=CHANNELS,
        rate=SEND_RATE,
        output=True,
        frames_per_buffer=1024,
    )
    if output_device is not None:
        spk_kwargs["output_device_index"] = output_device
    spk = pa.open(**spk_kwargs)

    url = f"{server_url}/stream?inp_lang={inp_lang}&opt_lang={opt_lang}"
    print(f"连接服务: {url}")

    async with websockets.connect(url) as ws:
        print("已连接，开始说话...")

        async def send_audio():
            while True:
                data = mic.read(FRAME_SIZE, exception_on_overflow=False)
                await ws.send(data)
                await asyncio.sleep(0.001)

        async def receive_audio():
            async for message in ws:
                spk.write(message)

        await asyncio.gather(send_audio(), receive_audio())


def list_devices():
    """列出所有音频输出设备。"""
    pa = pyaudio.PyAudio()
    print("\n可用音频输出设备:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            marker = " (默认)" if info.get("isDefaultOutput") else ""
            print(f"  [{i}] {info['name']}{marker}")
    print()
    pa.terminate()


def main():
    parser = argparse.ArgumentParser(
        description="实时语音翻译客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
示例:
  python client.py --list-devices
  python client.py --inp-lang zh --opt-lang en
  python client.py --inp-lang zh --opt-lang en --output-device 3
""",
    )
    parser.add_argument("--server", default="ws://localhost:8765", help="服务地址")
    parser.add_argument("--inp-lang", help="输入语言 (zh/en/ja/ko/fr/de/es)")
    parser.add_argument("--opt-lang", help="输出语言 (zh/en/ja/ko/fr/de/es)")
    parser.add_argument("--output-device", type=int, default=None, help="耳机设备编号")
    parser.add_argument("--list-devices", action="store_true", help="列出音频设备")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return

    if not args.inp_lang or not args.opt_lang:
        parser.error("需要指定 --inp-lang 和 --opt-lang")

    asyncio.run(run(args.server, args.inp_lang, args.opt_lang, args.output_device))


if __name__ == "__main__":
    main()
