# 调试记录

macOS (Apple Silicon M3) 环境下的问题排查与修复记录。

## 1. 依赖安装

### PyAudio 编译失败
- **现象**: `pip install pyaudio` 报错找不到 portaudio
- **修复**: `brew install portaudio`

### qwen-tts 缺少 onnxruntime
- **现象**: `import qwen_tts` 报 `ModuleNotFoundError: No module named 'onnxruntime'`
- **修复**: `pip install onnxruntime`

### qwen-tts 缺少 SoX
- **现象**: `import qwen_tts` 报 `SoX could not be found!`
- **修复**: `brew install sox`

### torchaudio 与 torch 版本不匹配
- **现象**: `import torchaudio` 报 `Symbol not found: _torch_library_impl`
- **原因**: vllm 将 torch 从 2.11.0 降级到 2.9.1，但 torchaudio 仍是 2.11.0
- **修复**: `pip install torchaudio==2.9.1`

### transformers 版本冲突 (Qwen3.5 vs qwen-tts)
- **现象**: transformers 4.x 不认识 `qwen3_5` 架构；升级到 5.x 后 qwen-tts 报 `check_model_inputs() missing 1 required positional argument`
- **原因**: Qwen3.5-4B 需要 transformers 5.x，但 qwen-tts 只兼容 4.x
- **修复**: 翻译模型从 Qwen3.5-4B 换为 Qwen3-1.7B（transformers 4.x 原生支持）

### Qwen3-1.7B-Instruct 不存在
- **现象**: HuggingFace 上没有 `Qwen/Qwen3-1.7B-Instruct` 这个仓库
- **原因**: Qwen3 系列 1.7B 只发布了 base 版本，没有 Instruct 版本
- **修复**: 使用 `Qwen/Qwen3-1.7B` base 模型，通过 system prompt 中的 `/no_think` 和明确指令约束输出格式

### huggingface-hub 版本残留
- **现象**: 降回 transformers 4.x 后报 `huggingface-hub>=0.34.0,<1.0 is required`
- **原因**: 之前升级 transformers 5.x 时 huggingface-hub 被拉到 1.8.0
- **修复**: `pip install "huggingface-hub>=0.34.0,<1.0"`

## 2. vLLM 不支持 macOS

### vLLM 引擎初始化失败
- **现象**: `RuntimeError: Engine core initialization failed` (warmup 阶段 crash)
- **原因**: vLLM 设计上依赖 CUDA GPU，macOS CPU fallback 不稳定
- **修复**: ASR 和 Translator 改为双后端架构——CUDA 环境用 vLLM，非 CUDA 用 Transformers

### ASR 流式识别不可用
- **现象**: `qwen_asr` 的 `init_streaming_state` 文档明确写 "Streaming ASR is supported ONLY for vLLM backend"
- **修复**: macOS 上改用 `Qwen3ASRModel.from_pretrained()` + 批量 `transcribe()`，按 chunk 攒够音频后识别

## 3. 运行时错误

### VAD 报 "Input audio chunk is too short"
- **现象**: 客户端连接后立即报 `ValueError: Input audio chunk is too short`
- **原因**: Silero VAD 在 16kHz 下要求最小 512 samples，但 FRAME_SIZE=480 (30ms) 不够
- **修复**: FRAME_SIZE 改为 512 (32ms)，客户端和服务端同步修改

### ASR transcribe 不接受 numpy array
- **现象**: `TypeError: Unsupported audio input type: <class 'numpy.ndarray'>`
- **原因**: `qwen_asr.transcribe()` 要求 `(ndarray, sample_rate)` 元组，不是裸 ndarray
- **修复**: `self.model.transcribe([(audio, SAMPLE_RATE)])`

### ASR 返回值不是 dict
- **现象**: `TypeError: 'ASRTranscription' object is not subscriptable`
- **原因**: `transcribe()` 返回 `ASRTranscription` dataclass，应用 `.text` 而非 `["text"]`
- **修复**: `results[0]["text"]` → `results[0].text`

### 翻译模型输出 `<think>` 内容
- **现象**: 翻译结果是 `<think>\nOkay,` 而非直接翻译
- **原因**: 使用了 base 模型 (Qwen3-4B)，未经指令微调，会自发进入推理模式
- **修复**: 换为更小的 Qwen3-1.7B，在 system prompt 中加 `/no_think` 禁用 thinking mode，`_clean_output()` 保留作为兜底清理残留 think 标签

### TTS 采样率不匹配
- **现象**: 耳机播放音调偏低、语速变慢
- **原因**: TTS 输出 24kHz，客户端按 16kHz 播放
- **修复**: pipeline.py 中用 `torchaudio.functional.resample()` 将 TTS 输出重采样到 16kHz

## 4. macOS 特有注意事项

### 设备自动检测
`config.py` 中 `DEVICE` 按 cuda → mps → cpu 优先级自动选择，各模块据此切换后端和 dtype。

### FP8 量化不可用
FP8 是 NVIDIA Ada/Hopper 架构专属，M 系列芯片不支持。macOS 上用 BF16 即可，1.7B 模型内存占用约 3.5GB。

### flash-attn 不可用
flash-attn 依赖 CUDA，macOS 上会 fallback 到 PyTorch 手动实现，性能略低但功能正常。