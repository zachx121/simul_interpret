       Perfect! 现在我有足够的信息来完成用户的请求。让我整理一份详细的事实汇报。

       ---
       详细模块分析报告

       1. ASR 模块 (/Users/bytedance/0-Code/YiSounda/models/asr.py)

       类名与构造签名：
       - 类名：StreamingASR（第 33 行）
       - 构造函数：__init__(self) 无参（第 35 行）

       依赖的 config 参数：
       - ASR_MODEL (第 19 行，config 第 30 行："Qwen/Qwen3-ASR-0.6B")
       - ASR_GPU_MEMORY_UTILIZATION (第 19 行，config 第 31 行：0.15)
       - ASR_MAX_NEW_TOKENS (第 19 行，config 第 35 行：32)
       - ASR_CHUNK_SIZE_SEC (第 19 行，config 第 32 行：2.0)
       - ASR_UNFIXED_CHUNK_NUM (第 22-23 行，config 第 33 行：2)
       - ASR_UNFIXED_TOKEN_NUM (第 22-23 行，config 第 34 行：5)
       - SENTENCE_ENDINGS (第 25 行，config 第 59 行：frozenset("。！？.!?"))
       - SAMPLE_RATE (第 26 行，config 第 21 行：16000)
       - DEVICE (第 27 行，config 第 14 行，由 _detect_device() 自动选择)

       对外提供的方法：

       1. start_session()（第 59-70 行）
         - 类型：初始化方法，不返回值
         - 用途：开始一个新的识别会话
       2. feed_audio(audio_chunk: np.ndarray) -> str | None（第 92-103 行）
         - 类型：流式增量识别
         - 输入：numpy array，单位是 samples（原始音频采样点）
         - 输出：当检测到句子标点时返回待翻译句子文本；否则返回 None
         - 流程：
             - vLLM 后端（CUDA）：逐帧流式识别，通过 streaming_transcribe() 推送到状态机（第 95 行）
           - Transformers 后端（非 CUDA）：攒音频帧到达阈值后批量识别（第 98-101 行）
       3. finish_session() -> str | None（第 72-88 行）
         - 类型：清理方法，返回剩余未提交的文本
         - 输出：尾部文本或 None

       macOS (非 CUDA) 与 CUDA 分支差异：

       ┌─────────────────┬─────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
       │      维度       │                   CUDA (vLLM)                   │              非 CUDA (Transformers)              │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ 初始化          │ LLM() 引擎（第 37-41 行）                       │ from_pretrained() + device_map（第 43-47 行）    │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ 识别模式        │ 逐帧流式，init_streaming_state()（第 66-70 行） │ 积累帧后批量识别（第 98-101 行）                 │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ feed_audio 返回 │ 可能在帧级返回已完成句子                        │ 等累积到阈值才识别一次                           │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ 结束会话        │ finish_streaming_transcribe()（第 75 行）       │ 一次识别剩余音频（第 81 行）                     │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ 识别 API        │ streaming_transcribe(chunk, state)（第 95 行）  │ transcribe([(audio, sr)]) 返回 list（第 110 行） │
       ├─────────────────┼─────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
       │ 文本来源        │ state.text（第 76 行）                          │ self._current_text（第 53 行）                   │
       └─────────────────┴─────────────────────────────────────────────────┴──────────────────────────────────────────────────┘

       内部缓冲与音频喂入方式：

       - vLLM 后端：
         - 无显式缓冲，每帧通过 streaming_transcribe() 推入状态机（第 95 行）
         - 状态机内部管理 unfixed_chunk_num=2 和 unfixed_token_num=5（第 67-68 行）
       - Transformers 后端：
         - 内部缓冲：self._audio_buffer: list[np.ndarray]（第 52 行）
         - 喂入方式：逐帧 append（第 98 行），当累积音频秒数 >= ASR_CHUNK_SIZE_SEC（2.0s）时触发识别（第 100-101 行）
       - 单位：样本数（samples），不是秒。调用者需按帧大小（FRAME_SIZE=512）来喂。

       标点驱动与中间/最终结果：

       - 中间结果标记：每次 feed_audio() 调用都检查 pending 文本中是否有句子标点（第 126-128 行）
       - 最终提交的标记：检测到 SENTENCE_ENDINGS（"。！？.!?"）的任何一个时，截取到最后一个标点位置（第 125-138 行）
       - 标点提交位置：在 asr.py 内部完成（第 121-138 行 _extract() 方法）
         - 返回值非 None 时表示"该句子已完整"
         - server.py 接收返回值后直接入翻译队列（server.py 第 116-117 行）
       - 标点提交时机：
         - 标点触发：当 pending 文本出现句子标点（第 115-116 行 vLLM，第 118-119 行 Transformers）
         - 静音兜底：在 server.py 中检测静音超时后调用 finish_session()（server.py 第 134-140 行）

       ---
       2. Translator 模块 (/Users/bytedance/0-Code/YiSounda/models/translator.py)

       类名与构造签名：
       - 类名：StreamingTranslator（第 18 行）
       - 构造函数：__init__(self) 无参（第 20 行）

       构造函数依赖的 config：
       - TRANSLATOR_MODEL (第 9 行，config 第 38 行："Qwen/Qwen3-1.7B")
       - TRANSLATOR_GPU_MEMORY_UTILIZATION (第 10 行，config 第 39 行：0.35)
       - TRANSLATOR_MAX_TOKENS (第 11 行，config 第 40 行：512)
       - DEVICE (第 12 行)

       对外方法：

       1. translate(text, src_lang, tgt_lang) -> str（第 45-52 行）
         - 类型：一次性同步翻译
         - 输入参数：
             - text: str 源语言文本
           - src_lang: str 源语言代码（如 "zh"）
           - tgt_lang: str 目标语言代码（如 "en"）
         - 输出：翻译后的完整文本字符串
         - 流程：
             - CUDA：调用 vLLM 的 llm.generate() 并返回完整结果（第 48-50 行）
           - 非 CUDA：调用 _translate_transformers() 使用 generate() API（第 52 行）
       2. translate_streaming(text, src_lang, tgt_lang)（第 54-61 行）
         - 类型：流式翻译，使用 Python yield 返回逐步累积的文本
         - 输入参数：同 translate()
         - 输出：每次 yield 返回累积到当前时刻的完整翻译文本（而非增量）
         - 流程：
             - CUDA：vLLM 的流式生成，逐次 yield 累积文本（第 58-59 行）
           - 非 CUDA：调用 _translate_streaming_transformers()（第 61 行）

       System Prompt 中的 /no_think 与输出清理：

       - /no_think 位置：在 _build_messages() 方法（第 119-128 行）
         - 第 123 行：system role 的 content 首行是 "/no_think\n"
         - 作用：禁用 Qwen3 的 thinking mode（避免输出 <think>...</think> 块）
       - 输出清理逻辑：_clean_output(text: str) -> str（第 130-136 行）
         - 检查是否存在 "<think>" 标签（第 132 行）
         - 若存在，查找 "</think>" 的位置，截取其后内容（第 133-135 行）
         - 最后 .strip() 去空白（第 136 行）
         - 调用位置：
             - vLLM：每次 yield 时调用（第 59 行）
           - Transformers：每次 yield 时调用（第 107 行）

       输入参数细节：

       在 _build_messages()（第 119-128 行）和 _build_prompt()（第 111-117 行）中，语言参数的传递：
       - src_lang 和 tgt_lang 直接用在 prompt/messages 中
       - 需配合 config.py 的 LANG_MAP（config 第 47-55 行）转换
       - pipeline.py 第 53-54 行负责这个映射：src_lang = LANG_MAP[inp_lang]

       ---
       3. TTS 模块 (/Users/bytedance/0-Code/YiSounda/models/tts.py)

       类名与构造签名：
       - 类名：TTSEngine（第 17 行）
       - 构造函数：__init__(self) 无参（第 19 行）

       synthesize 方法：

       def synthesize(self, text: str, language: str) -> tuple[np.ndarray, int]:
       （第 28-43 行）

       - 是否流式：否，逐子句调用（整段合成后返回）
       - 调用位置：pipeline.py 第 98 行（self.tts.synthesize(text, language)）
       - 返回值：(audio_array, sample_rate)
         - audio_array：float32 numpy array（第 43 行 wavs[0]）
         - sample_rate：24000（DEBUG.md 第 72-75 行提到 "TTS 输出 24kHz"，pipeline.py 第 101-104 行重采样到 SAMPLE_RATE=16000）

       Language 参数取值范围：

       - 来源：config.py LANG_MAP 的 value（config 第 47-55 行）
         - "Chinese" (zh)
         - "English" (en)
         - "Japanese" (ja)
         - "Korean" (ko)
         - "French" (fr)
         - "German" (de)
         - "Spanish" (es)
       - pipeline.py 第 54 行：tgt_lang = LANG_MAP[opt_lang] 传入 synthesize()

       ---
       4. VAD 模块 (/Users/bytedance/0-Code/YiSounda/models/vad.py)

       类名与构造签名：
       - 类名：VoiceDetector（第 10 行）
       - 构造函数：__init__(self, threshold: float = 0.5)（第 12 行）
         - 默认阈值：0.5（DEBUG.md 第 55 行提到 "Silero VAD 在 16kHz 下要求最小 512 samples"）

       接口：

       def is_speech(self, audio_chunk: torch.Tensor) -> bool:
       （第 18-25 行）

       - 输入：
         - audio_chunk：float32 torch.Tensor，16kHz mono
         - 长度：512 samples（32ms，DEBUG.md 第 55 行）
         - 对应 FRAME_SIZE=512（config 第 23 行）
       - 输出：bool，表示该帧是否包含人声
         - 内部比较：prob > self.threshold（第 25 行）

       其他方法：

       def reset(self):
       （第 27-29 行）
       - 重置 VAD 内部状态，在一段语音结束后调用
       - 调用位置：server.py 第 144 行（静音超时时调用）

       ---
       5. Pipeline 模块 (/Users/bytedance/0-Code/YiSounda/pipeline.py)

       类名与编排逻辑：
       - 类名：StreamingPipeline（第 28 行）

       翻译 → TTS 编排方式：

       def process_sentence(self, text, inp_lang, opt_lang, on_audio_chunk):
       （第 35-88 行）

       - 是否边翻译边合成：是（实现了流式翻译与 TTS 的并行处理）
         - 流式翻译循环：for partial_text in self.translator.translate_streaming(...)（第 66-82 行）
         - 每次 yield 到达时，检查新增部分是否包含子句分隔符（第 72-76 行）
         - 一旦凑够一个子句就立即调用 _synthesize_and_send() 合成并发送（第 81 行）
         - 翻译尾部无标点时也会合成（第 85-87 行）

       子句切分逻辑：

       位置：pipeline.py 第 72-82 行
       - 分隔符：TTS_SPLIT_DELIMITERS（config 第 63 行 = SENTENCE_ENDINGS | CLAUSE_DELIMITERS）
         - 句子标点："。！？.!?"
         - 子句标点："，,、:：;；"
       - 切分算法：
         a. 追踪 sent_so_far（已处理的累积文本）和 accumulated（当前翻译总文本）
         b. 计算新增部分：new_part = accumulated[len(sent_so_far):]（第 70 行）
         c. 在新增部分中找最后一个分隔符位置（第 73-76 行）
         d. 若找到，截取到分隔符为止（new_part[: last_delim_pos + 1]），清理后合成（第 79-81 行）
         e. 更新 sent_so_far 指针（第 82 行）

       日志与可观测性：

       - 第 55 行：log.info(f"🔄 开始处理: ...")
       - 第 89 行：log.info(f"✅ 翻译完成: ...")
       - 第 106 行：log.info(f"🔊 TTS: ...") 包含时长信息

       ---
       6. Server 模块 (/Users/bytedance/0-Code/YiSounda/server.py)

       WebSocket 协议与帧格式：

       - 协议文档：第 44-59 行的 @app.websocket("/stream") 装饰器下的 docstring
         - 客户端发送：PCM 16-bit LE, 16 kHz, mono
         - 每帧大小：512 样本 = 32ms（DEBUG.md 第 55 行）
         - 格式：二进制（await ws.send_bytes(pcm)，client.py 第 68 行）
       - 帧接收与转换（第 95-99 行）：
       data = await ws.receive_bytes()  # 二进制 PCM 16-bit
       frame = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
       tensor = torch.tensor(frame)

       双触发架构：

       位置：server.py 第 92-145 行的主循环

       ┌──────────┬─────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────────────────────┐
       │  触发器  │                                条件                                 │                            处理                             │
       ├──────────┼─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
       │ 标点触发 │ vad.is_speech() 为真 AND asr.feed_audio() 返回非 None（检测到句点） │ 第 115-117 行：入队翻译                                     │
       ├──────────┼─────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────┤
       │ 静音兜底 │ is_speaking 为真且 silence_count >= SILENCE_FRAMES                  │ 第 133-140 行：调用 finish_session() 获取剩余文本，入队翻译 │
       └──────────┴─────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────────────────────┘

       具体流程：

       1. 人声开始（第 101-109 行）：
         - VAD 检测 is_speech() 为真且当前非说话状态
         - 创建 ASR 会话 start_session()
       2. 人声期间（第 111-117 行）：
         - 持续喂帧 feed_audio()
         - 若返回非 None（标点触发），入队
       3. 静音期间（第 119-145 行）：
         - 继续喂帧保持 ASR 连贯性（第 123-129 行）
         - 累计静音帧数（第 120 行）
         - 当静音超过阈值时（第 132 行）：
             - 调用 finish_session() 获取尾部文本（第 134-135 行）
           - 若有剩余，入队翻译（第 138-140 行）
           - 重置状态（第 142-144 行）

       ASR 与 Translator 衔接：

       - 数据流向：ASR 返回的句子 → queue.put(sentence)（server.py 第 117, 129, 140 行）
       - Translation Worker（第 77-88 行）：从队列顺序取句子，调用 pipeline.process_sentence()
       - 顺序保证：单线程 worker，确保播放顺序 = 说话顺序

       日志点：

       - 第 107 行：人声开始
       - 第 116 行：标点触发
       - 第 128 行：静音中标点触发
       - 第 139 行：静音兜底
       - 第 145 行：会话结束

       ---
       7. Client 模块 (/Users/bytedance/0-Code/YiSounda/client.py)

       麦克风读取与帧发送：

       位置：client.py 第 30-75 行

       - 麦克风初始化（第 39-45 行）：
       mic = pa.open(
           format=FORMAT,       # pyaudio.paInt16
           channels=CHANNELS,   # 1
           rate=SEND_RATE,      # 16000
           input=True,
           frames_per_buffer=FRAME_SIZE,  # 512
       )
       - 帧读取与发送（第 65-69 行）：
       async def send_audio():
           while True:
               data = mic.read(FRAME_SIZE, exception_on_overflow=False)
               await ws.send(data)  # 直接发送 PCM 16-bit bytes
               await asyncio.sleep(0.001)
         - 每次读 512 samples（32ms）
         - 直接发送二进制，无需显式转换（PyAudio 已返回 bytes）
       - 耳机播放（第 47-57 行 初始化，第 71-73 行 接收并播放）：
       spk = pa.open(...)  # 初始化耳机输出

       async def receive_audio():
           async for message in ws:
               spk.write(message)  # 接收 PCM bytes 直接写入耳机

       文件输入模式 (client_2.py)：

       - 第 106 行：--list-devices 列出设备
       - 没有直接的 "读 WAV 文件模拟麦克风" 开关
       - 但 client_2.py 提供了文件输入的替代方案（第 24-134 行）：
         - --input 参数：指定音频文件路径（第 148 行）
         - 内部流程：
             - 读取文件→重采样到 16kHz mono（第 31-45 行）
           - 按 FRAME_SIZE 切分，模拟实时速率发送（第 61-80 行）
           - 接收翻译音频并保存（第 91-134 行）

       ---
       8. 测试基础设施 (/Users/bytedance/0-Code/YiSounda/tests/)

       当前测试文件：

       - /Users/bytedance/0-Code/YiSounda/tests/gen_test_audio.py（第 1-74 行）
         - 用途：生成固定文本的测试音频
         - 输入：TEXT = "你好，我正在新加坡与你通话。"（第 26 行）
         - 输出：
             - /Users/bytedance/0-Code/YiSounda/tests/fixtures/hello_singapore_zh.wav（16kHz mono PCM16）
           - /Users/bytedance/0-Code/YiSounda/tests/fixtures/hello_singapore_zh.txt（参考文本）
       - /Users/bytedance/0-Code/YiSounda/tests/fixtures/hello_singapore_zh.wav 和 .txt 已存在

       requirements.txt 中的测试相关依赖：

       - 无 pytest（requirements.txt 第 1-17 行未包含）
       - 现有依赖：numpy, torch, torchaudio, soundfile, fastapi, uvicorn, qwen-asr, qwen-tts, vllm, websockets, pyaudio

       现成的测试脚本或 mock：

       - 无现成的单元测试框架
       - gen_test_audio.py 是唯一的测试数据生成脚本

       ---
       9. 「流式 / 实时」的可观测信号

       当前代码的时间戳记录现状：

       ┌───────────────────────────┬────────────────────────────────┬────────────────────────────────────────┬───────────────────────────────────────────────
       ────┐
       │           模块            │            可观测点            │                具体位置                │                    获取的信息
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ ASR (asr.py)              │ feed_audio() 返回非 None       │ asr.py 第 115-120 行                   │ 句子标点检测时刻
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ finish_session() 返回非 None   │ asr.py 第 72-88 行                     │ 静音兜底的尾部文本
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ 无中间结果 yield               │ -                                      │ 缺乏帧级可观测信号
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ Translator                │ translate_streaming() yield    │ translator.py 第 54-61 行              │ 流式翻译的累积文本，可在 pipeline.py 第 66 行
           │
       │ (translator.py)           │                                │                                        │ hook
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ 无时间戳                       │ -                                      │ yield 时无时间信息
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ TTS (tts.py)              │ synthesize() 返回的 sr         │ tts.py 第 43 行                        │ 采样率信息
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ 无流式输出                     │ -                                      │ 整段合成后返回
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ Pipeline (pipeline.py)    │ 循环收集 partial_text          │ pipeline.py 第 66-82 行                │ 流式翻译边界检测
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ log.info() 调用                │ pipeline.py 第 55, 89, 106 行          │ 处理开始、翻译完成、TTS 合成时长
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ Server (server.py)        │ log.info() 调用                │ server.py 第 107, 116, 128, 139, 145   │ VAD 事件、标点触发、静音兜底时刻
           │
       │                           │                                │ 行                                     │
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │                           │ silence_count 和               │ server.py 第 40-41, 64, 120, 132 行    │ 静音累积帧数
           │
       │                           │ SILENCE_FRAMES                 │                                        │
           │
       ├───────────────────────────┼────────────────────────────────┼────────────────────────────────────────┼───────────────────────────────────────────────
       ────┤
       │ Client (client.py)        │ WebSocket 发送/接收            │ client.py 第 68, 73 行                 │ 发送帧、接收音频事件
           │
       └───────────────────────────┴────────────────────────────────┴────────────────────────────────────────┴───────────────────────────────────────────────
       ────┘

       Hook 点（可插入时间戳的位置）：

       1. ASR 流式识别首个句子的延迟：
         - Hook 位置：asr.feed_audio() 返回首个非 None 时
         - 对应 server.py 第 112-117 行
         - 计时：从 start_session() 到第一个标点触发
       2. Translator 流式翻译增量产出：
         - Hook 位置：translator.translate_streaming() 的每个 yield（第 54-61 行）
         - pipeline.py 第 66 行循环可注入时间戳
         - 计时：每次 yield 的时间间隔
       3. TTS 子句级合成延迟：
         - Hook 位置：_synthesize_and_send() 调用前后（pipeline.py 第 91-108 行）
         - 计时：翻译完成→TTS 合成完成的延迟
       4. 服务端端到端延迟：
         - Hook 位置：server.py 第 116 行（标点触发）→ pipeline.py 第 89 行（翻译完成）
         - 计时：标点触发时刻→翻译完成时刻
       5. 客户端首字延迟：
         - Hook 位置：client.py 第 73 行 spk.write(message) 接收首个音频块时
         - 计时：发送首个音频帧→接收首个翻译音频的时间差

       现有日志系统：

       - server.py 第 26-30 行：logging 初始化，格式包含时间戳 "%(asctime)s"
       - pipeline.py 第 25 行：log = logging.getLogger("pipeline")
       - 可通过 logging.StreamHandler 或自定义 Formatter 获取毫秒级时间戳

       缺失的可观测信号：

       - ASR 内部状态（vLLM 状态机的 state.text 增量）
       - Translator 模块内的 token 级时间戳（仅在 pipeline.py 层级能观测）
       - TTS 内部流式进度（当前只支持整段合成）
       - 客户端音频缓冲/延迟信息

       ---
       总结表格

       ┌───────────┬─────────────────────────────┬──────────────────────┬──────────────────────────────┬──────────────────────────────┬──────────────────────
       ───┐
       │   模块    │          文件路径           │     关键方法签名     │            流式？            │         时间戳 hook          │         需补充
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ VAD       │ /models/vad.py:10-29        │ is_speech(tensor) -> │ 否（帧级但同步）             │ VAD 决策时刻（第 101 行）    │ -
          │
       │           │                             │  bool                │                              │                              │
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ ASR       │ /models/asr.py:33-139       │ feed_audio(chunk) -> │ 是（vLLM）/否（Transformers  │ 标点检测→queue.put（server.p │ ASR 增量识别进度
          │
       │           │                             │  str|None            │ ）                           │ y 116 行）                   │
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ Translato │ /models/translator.py:18-13 │ translate_streaming( │ 是（均支持）                 │ 每个 yield 时刻（pipeline.py │ 翻译速度度量
          │
       │ r         │ 7                           │ ) -> yield str       │                              │  66 行）                     │
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ TTS       │ /models/tts.py:17-43        │ synthesize(text,     │ 否                           │ 合成完成时刻（pipeline.py    │
       首包延迟（当前为子句级  │
       │           │                             │ lang) -> (array, sr) │                              │ 106 行）                     │ ）
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ Pipeline  │ /pipeline.py:28-109         │ process_sentence(... │ 是（子句级）                 │ log.info（55, 89, 106 行）   │ 子句边界时刻
          │
       │           │                             │ )                    │                              │                              │
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ Server    │ /server.py:44-149           │ stream_endpoint()    │ 是（双触发）                 │ log.info（107, 116, 128,     │ 帧级时间戳
          │
       │           │                             │                      │                              │ 139, 145 行）                │
          │
       ├───────────┼─────────────────────────────┼──────────────────────┼──────────────────────────────┼──────────────────────────────┼──────────────────────
       ───┤
       │ Client    │ /client.py:30-120           │ run(...)             │ 是（实时收发）               │ 接收音频事件（73 行）        │ 音频块到达时间
          │
       └───────────┴─────────────────────────────┴──────────────────────┴──────────────────────────────┴──────────────────────────────┴──────────────────────
       ───┘
  
