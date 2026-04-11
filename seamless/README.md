# SeamlessStreaming Demo

最小流式同传 demo，用来验证 `facebook/seamless-streaming` 在本机能跑通。

## 文件

- `demo.py` — 一个脚本两种模式，共享同一套流式音频输入
  - `stream_wav_chunks(path, cap_sec)` — 按 320ms 切块流式产出 PCM
  - `run(mode="s2tt", ...)` — 流式打印英文翻译文本
  - `run(mode="s2st", ...)` — 流式收集英文翻译音频，存到 `out.wav`

## 安装（只列 Seamless 相关的依赖，假设项目已有 torch/numpy/soundfile）

```bash
pip install fairseq2            # macOS ARM64 和 Linux x86-64 有预编译 wheel
pip install seamless_communication
pip install simuleval            # 流式推理框架，seamless agents 基于它
```

首次运行会从 HuggingFace 下载：

- `seamless_streaming_unity`（speech encoder + NAR T2U + vocoder）
- `seamless_streaming_monotonic_decoder`（EMMA 文本解码器）

合计约 10 GB。

## 运行

```bash
# 流式文本翻译（中 → 英，只读前 8 秒音频）
python seamless/demo.py --mode s2tt --tgt-lang eng --cap 8

# 流式语音翻译，写到 out.wav
python seamless/demo.py --mode s2st --tgt-lang eng --cap 8 --out out.wav
```

`--tgt-lang` 用 ISO 639-3：`eng / cmn / jpn / kor / fra / deu / spa`。

## 核心要点（对上下游的简要说明）

- 入：`SpeechSegment(content=list(float), sample_rate=16000, tgt_lang=..., finished=False)`
- 出：`pushpop(seg)` 返回一个 `Segment`
  - `data_type == "text"` 且 `content` 非空 → 本轮吐出的翻译文本片段
  - `data_type == "speech"` → 一段合成好的 PCM（`s2st` 模式）
  - `is_empty == True` 或 `finished == True` → 本轮没有新产出
- 推完最后一块音频要补一个 `EmptySegment(finished=True)` 让 pipeline 知道源结束，并继续 `pop()` 把尾部吐干净。

## 核心超参数

写在 `build_agent()` 里的 `argv`，和 Seamless 官方 CLI 默认一致：

| 参数 | 值 | 含义 |
|---|---|---|
| `--source-segment-size` | 320 | 每块音频 ms |
| `--decision-threshold` | 0.5 | EMMA 文本 WRITE 阈值，调小延迟更低、BLEU 略降 |
| `--min-starting-wait-w2vbert` | 192 | 起步前至少积累多少帧才开始解码 |
| `--min-unit-chunk-size` | 50 | **只 s2st 用**：unit 攒到多少才送 vocoder 合成语音 |
| `--max-len-b` | 100 | 生成长度上限相关 |

## 已知待验证

- [ ] macOS / MPS 下 `fairseq2` 权重加载是否成功（Apple Silicon 有 wheel 但算子覆盖可能不全）
- [ ] `pushpop` 在源未结束时返回空 segment 的具体行为（论文 Algorithm 1 要求源未结束时若提前 finished 应重置；`UnitYAgentPipeline.pop` 已处理，无需手工管）
- [ ] 流式音频 chunk 的采样率是否始终 = 16000（vocoder 输出 16k，和输入一致，直接 concat 保存）
