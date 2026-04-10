"""
FastAPI WebSocket 服务 — 标点驱动 + 静音兜底的双触发流式翻译。

数据流：
  Client → [PCM 16-bit 帧] → Server
  Server → VAD → Streaming ASR → (标点触发 / 静音兜底) → 翻译队列
  translation_worker: 队列 → 流式翻译 → 子句 TTS → [PCM bytes] → Client

关键设计：
  · ASR feed_audio() 返回非 None 时表示检测到句子标点，立即入队
  · translation_worker 单线程顺序处理，保证播放顺序 = 说话顺序
  · 翻译+TTS 在 executor 中运行，不阻塞音频帧接收
"""

import asyncio
import logging

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, Query

from config import HOST, PORT, FRAME_SIZE, VAD_THRESHOLD, SILENCE_TIMEOUT_MS

log = logging.getLogger("server")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
from models.vad import VoiceDetector
from pipeline import StreamingPipeline

app = FastAPI(title="Voice Translation Service")

# 全局模型（进程启动时加载一次）
vad = VoiceDetector(threshold=VAD_THRESHOLD)
pipeline = StreamingPipeline()

# 静音帧数阈值
SILENCE_FRAMES = int(SILENCE_TIMEOUT_MS / (1000 / (16000 / FRAME_SIZE)))


@app.websocket("/stream")
async def stream_endpoint(
    ws: WebSocket,
    inp_lang: str = Query(..., description="输入音频语言 (zh/en/ja/ko/fr/de/es)"),
    opt_lang: str = Query(..., description="输出音频语言 (zh/en/ja/ko/fr/de/es)"),
):
    """
    WebSocket 流式翻译端点。

    连接方式:
        ws://host:port/stream?inp_lang=zh&opt_lang=en

    协议:
        客户端发送 → PCM 16-bit LE, 16 kHz, mono, 每帧 480 样本 (30ms)
        服务端发送 → PCM 16-bit LE, 翻译合成后的音频片段
    """
    await ws.accept()
    loop = asyncio.get_event_loop()

    # ── 状态 ──────────────────────────────────
    silence_count = 0
    is_speaking = False
    asr_active = False

    # ── 翻译队列 + worker ─────────────────────
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def _ws_send(pcm: bytes):
        await ws.send_bytes(pcm)

    def _sync_send(pcm: bytes):
        asyncio.run_coroutine_threadsafe(_ws_send(pcm), loop)

    async def _translation_worker():
        """从队列顺序取句子 → 翻译 → TTS → 发回客户端。"""
        while True:
            sentence = await queue.get()
            if sentence is None:
                break
            await loop.run_in_executor(
                None,
                pipeline.process_sentence,
                sentence, inp_lang, opt_lang, _sync_send,
            )
            queue.task_done()

    worker = asyncio.create_task(_translation_worker())

    # ── 帧处理主循环 ─────────────────────────
    try:
        while True:
            data = await ws.receive_bytes()

            # PCM 16-bit → float32
            frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.tensor(frame)

            if vad.is_speech(tensor):
                silence_count = 0

                # 人声开始 → 创建 ASR 会话
                if not is_speaking:
                    is_speaking = True
                    log.info("🎤 检测到人声，开始 ASR 会话")
                    await loop.run_in_executor(None, pipeline.asr.start_session)
                    asr_active = True

                # 喂帧 + 检查标点触发
                sentence = await loop.run_in_executor(
                    None, pipeline.asr.feed_audio, frame,
                )
                if sentence:
                    log.info(f"📝 ASR 标点触发: \"{sentence}\"")
                    await queue.put(sentence)

            elif is_speaking:
                silence_count += 1

                # 静音期间继续喂帧（保持 ASR 连贯性）
                if asr_active:
                    sentence = await loop.run_in_executor(
                        None, pipeline.asr.feed_audio, frame,
                    )
                    if sentence:
                        log.info(f"📝 ASR 标点触发(静音中): \"{sentence}\"")
                        await queue.put(sentence)

                # 静音超时 → 兜底提交
                if silence_count >= SILENCE_FRAMES:
                    if asr_active:
                        remaining = await loop.run_in_executor(
                            None, pipeline.asr.finish_session,
                        )
                        asr_active = False
                        if remaining:
                            log.info(f"📝 ASR 静音兜底: \"{remaining}\"")
                            await queue.put(remaining)

                    silence_count = 0
                    is_speaking = False
                    vad.reset()
                    log.info("🔇 静音超时，会话结束")

    finally:
        await queue.put(None)
        await worker


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
