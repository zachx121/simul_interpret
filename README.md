# 实时流式语音翻译服务

基于 Qwen3 系列模型的端到端流式语音翻译：说中文，耳机里听到英文——句子说到一半翻译就已经开始了。

## 特性

- **全链路流式**：ASR 边听边识别 → 标点即时触发翻译 → 翻译逐子句合成播放
- **标点驱动提交**：不等整段话说完，每识别到一个句子（句号/问号/感叹号）立即送翻译
- **低延迟**：首句翻译音频在句子说完后 ~0.3-0.6s 到达耳机
- **7 语言**：中文、英文、日文、韩文、法文、德文、西班牙文

## 模型组件

| 组件 | 模型 | 大小 | 运行位置 |
|------|------|------|----------|
| VAD | Silero VAD v5 | 2 MB | CPU |
| ASR | Qwen3-ASR-0.6B | 1.9 GB | GPU / MPS / CPU |
| 翻译 | Qwen3-1.7B (text-only) | 3.5 GB | GPU / MPS / CPU |
| TTS | Qwen3-TTS-12Hz-1.7B-CustomVoice | 3.5 GB | GPU / MPS / CPU |

CUDA GPU 合计约 14 GB，一张 RTX 4090 / RTX 3090 / A5000 即可。macOS (Apple Silicon) 也可通过 MPS 后端运行。

## 安装

```bash
# 建议使用 conda 创建隔离环境
conda create -n voice-translate python=3.12 -y
conda activate voice-translate

# 安装依赖
pip install -r requirements.txt

# 系统依赖（PyAudio + SoX）：
# Ubuntu: sudo apt-get install python3-pyaudio portaudio19-dev sox
# macOS:  brew install portaudio sox
```

模型权重会在首次运行时自动从 HuggingFace 下载。如需手动预下载：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-1.7B
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
huggingface-cli download Qwen/Qwen3-TTS-Tokenizer-12Hz
```

## 使用

### 启动服务端

```bash
python server.py
# 监听 ws://0.0.0.0:8765/stream
```

### 启动客户端

```bash
# 1. 列出音频设备，找到耳机编号
python client.py --list-devices

# 2. 开始翻译（中文 → 英文，耳机设备编号 3）
python client.py --inp-lang zh --opt-lang en --output-device 3

# 其他语言组合
python client.py --inp-lang en --opt-lang ja      # 英 → 日
python client.py --inp-lang fr --opt-lang zh      # 法 → 中
```

## 架构

```
Client                          Server
──────                          ──────
Mic → 30ms PCM ─── WS ───→ VAD → Streaming ASR
                                    │
                            检测到句号/问号/感叹号？
                            ├─ 是 → 句子入翻译队列（用户继续说话）
                            └─ 否 → 继续累积
                                    │
                            静音 500ms？
                            └─ 是 → 尾部文本入翻译队列
                                    │
                            translation_worker（顺序处理）
                                    │
                            流式翻译 → 子句切分 → TTS
                                    │
Speaker ← PCM ←── WS ────← 合成音频
```

## 配置

所有参数集中在 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `VAD_THRESHOLD` | 0.5 | 人声判定阈值，降低可更灵敏 |
| `SILENCE_TIMEOUT_MS` | 500 | 静音多久视为一段话结束 |
| `ASR_CHUNK_SIZE_SEC` | 2.0 | ASR 内部 chunk，影响识别延迟 |
| `TTS_SPEAKER` | "Vivian" | TTS 说话人，可选值见 Qwen3-TTS 文档 |

## 项目结构

```
voice_translate_service/
├── server.py          # WebSocket 服务（双触发架构）
├── client.py          # 麦克风 + 耳机客户端
├── config.py          # 全局配置
├── pipeline.py        # 翻译 → TTS 编排
├── models/
│   ├── vad.py         # Silero VAD
│   ├── asr.py         # 流式 ASR + 标点提交
│   ├── translator.py  # 流式翻译
│   └── tts.py         # 语音合成
├── requirements.txt
└── README.md
```

## 支持语言

| 代码 | 语言 |
|------|------|
| zh | 中文 |
| en | 英文 |
| ja | 日文 |
| ko | 韩文 |
| fr | 法文 |
| de | 德文 |
| es | 西班牙文 |

## 已知限制

- 需要佩戴耳机（否则 TTS 输出会被麦克风再次捕捉形成回路）
- 单用户设计（首期不支持并发连接）
- TTS 采样率可能与客户端播放采样率不完全匹配，如遇音调异常请在 `config.py` 中调整 `SAMPLE_RATE`
