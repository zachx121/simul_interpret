"""三级流式 Pipeline：ASR → 翻译 → TTS

本模块负责单个句子的处理：流式翻译 → 子句切分 → TTS 合成 → 回调发送。

调用方（server.py）负责：
  - 通过 ASR 标点检测或静音超时，将句子逐个提交到 process_sentence
  - 通过 asyncio.Queue 保证句子间的顺序

process_sentence 在翻译输出中监测子句标点（逗号、句号等），
一旦凑出一个子句就立即送入 TTS 合成并回调发送音频，
实现翻译与 TTS 的并行。
"""

import logging
import numpy as np
import torchaudio.functional as F
import torch
from typing import Callable

from models.asr import StreamingASR
from models.translator import StreamingTranslator
from models.tts import TTSEngine
from config import LANG_MAP, TTS_SPLIT_DELIMITERS, SAMPLE_RATE

log = logging.getLogger("pipeline")


class StreamingPipeline:

    def __init__(self):
        self.asr = StreamingASR()
        self.translator = StreamingTranslator()
        self.tts = TTSEngine()

    def process_sentence(
        self,
        text: str,
        inp_lang: str,
        opt_lang: str,
        on_audio_chunk: Callable[[bytes], None],
    ):
        """处理单个句子：流式翻译 → 子句 TTS → 回调发送音频。

        此方法可能在用户还在说话时被调用（标点触发），
        也可能在静音超时后被调用（兜底触发）。

        Args:
            text: ASR 识别出的源语言句子
            inp_lang: 输入语言代码（如 "zh"）
            opt_lang: 输出语言代码（如 "en"）
            on_audio_chunk: 回调，每合成一段音频调用一次，参数为 PCM bytes
        """
        src_lang = LANG_MAP[inp_lang]
        tgt_lang = LANG_MAP[opt_lang]
        log.info(f"🔄 开始处理: \"{text}\" ({src_lang} → {tgt_lang})")

        # 输入输出语言相同 → 跳过翻译，直接合成
        if inp_lang == opt_lang:
            self._synthesize_and_send(text, tgt_lang, on_audio_chunk)
            return

        # 流式翻译 + 子句切分 + TTS
        sent_so_far = ""
        accumulated = ""

        for partial_text in self.translator.translate_streaming(
            text, src_lang, tgt_lang
        ):
            accumulated = partial_text
            new_part = accumulated[len(sent_so_far) :]

            # 在新增部分中找最后一个子句分隔符
            last_delim_pos = -1
            for i, ch in enumerate(new_part):
                if ch in TTS_SPLIT_DELIMITERS:
                    last_delim_pos = i

            if last_delim_pos >= 0:
                clause = new_part[: last_delim_pos + 1].strip()
                if clause:
                    self._synthesize_and_send(clause, tgt_lang, on_audio_chunk)
                sent_so_far = accumulated[: len(sent_so_far) + last_delim_pos + 1]

        # 处理翻译尾部（可能没有标点结尾）
        remaining = accumulated[len(sent_so_far) :].strip()
        if remaining:
            self._synthesize_and_send(remaining, tgt_lang, on_audio_chunk)

        log.info(f"✅ 翻译完成: \"{text}\" → \"{accumulated}\"")

    def _synthesize_and_send(
        self,
        text: str,
        language: str,
        on_audio_chunk: Callable[[bytes], None],
    ):
        """合成一段文本并通过回调发送 PCM 音频。"""
        audio, sr = self.tts.synthesize(text, language)
        if len(audio) > 0:
            # TTS 采样率可能与协议采样率不同（如 24kHz vs 16kHz），需重采样
            if sr != SAMPLE_RATE:
                audio = F.resample(
                    torch.from_numpy(audio), orig_freq=sr, new_freq=SAMPLE_RATE
                ).numpy()
            duration = len(audio) / SAMPLE_RATE
            log.info(f"🔊 TTS: \"{text}\" → {duration:.2f}s 音频")
            pcm_bytes = (audio * 32768).astype(np.int16).tobytes()
            on_audio_chunk(pcm_bytes)
