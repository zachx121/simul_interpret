# Plan: YiSounda 分模块自动化测试

## Context

项目 `/Users/bytedance/0-Code/YiSounda` 是基于 Qwen3 系列的实时流式语音翻译服务。在当前 M3 Mac 上整条链路（server+client）还没能跑通，直接启动调试粒度太粗。目的是为 ASR / Translator / TTS 三个 ML 模块分别建立**可重复、带期望输出、带 pass/fail 退出码**的单模块测试脚本，方便在遇到 bug 时端到端自动迭代（运行→读失败→修→再运行），直到每个模块独立跑通，再做端到端集成。

已完成：
- `tests/gen_test_audio.py`（TTS 合成 fixture 生成器）
- `tests/fixtures/hello_singapore_zh.wav`（16kHz mono PCM16, 3.12s）
- `tests/fixtures/hello_singapore_zh.txt`（参考文本 `你好，我正在新加坡与你通话。`）

测试顺序（按用户指定）：**ASR → Translator → TTS**。不做 pipeline 端到端测试。

Python 解释器统一使用 `/opt/homebrew/Caskroom/miniconda/base/envs/yisounda/bin/python`。

---

## 关键背景事实（探索后确认）

- `models/asr.py:33` `StreamingASR` 在 macOS（非 CUDA）走 Transformers 分支：`feed_audio()` 会把帧累积到 `ASR_CHUNK_SIZE_SEC=2.0s` 才批量 `transcribe()`。所以在这台 M3 上"流式"≈2 秒 chunk 批处理。
- `models/asr.py:92` `feed_audio(audio_chunk: np.ndarray) -> str | None`：逐帧喂入，返回值非 None 时表示"已检测到句子标点并截出一整句"。尾部剩余靠 `finish_session()`（`asr.py:72`）收回。
- 标点判定在 `asr.py:121` `_extract()` 内部完成，用 `SENTENCE_ENDINGS`（`config.py:59`）。
- `models/translator.py:18` `StreamingTranslator.translate_streaming(text, src_lang, tgt_lang)`：yield **累积** 文本（非增量片段）。`/no_think` 和 `<think>` 清理在 `translator.py:119` 和 `translator.py:130`。
- `models/tts.py:17` `TTSEngine.synthesize(text, language)`：非流式，返回 `(float32 ndarray, 24000)`。`language` 取 `LANG_MAP.values()`（`config.py:47`），如 `"Chinese"` / `"English"`。
- `FRAME_SIZE=512`（`config.py:23`）、`SAMPLE_RATE=16000`（`config.py:21`）。
- requirements.txt 无 pytest；每个测试写成可独立运行的 `__main__` 脚本，退出码标识成功/失败。

---

## 设计决策

- **Translator 正确性**：关键词组命中，参考放 `tests/fixtures/translations_ref.json`，方便后续手工调。
- **TTS 测试**：做 ASR round-trip（合成→再识别→CER 与原文对比），两边同时校验。
- **"流式"自动断言**：只对 Translator 断言 `len(yields) ≥ 2` 且 `len(yields[i])` 单调递增。ASR（macOS batch）和 TTS（非流式）不做流式断言，只做正确性。
- **ASR 正确性指标**：字符级 CER（`difflib.SequenceMatcher`，归一化去标点/空白），阈值 `CER < 0.15`。
- **实时性**：不做 RTF 硬断言（避免 macOS 首次加载慢导致 flakey），但仍记录 wall-clock 到日志便于诊断。

---

## 要创建的文件

| 路径 | 作用 |
|------|------|
| `tests/_utils.py` | 共享工具：load_wav / save_wav / cer / normalize / frame 切分 / TimestampLog |
| `tests/test_asr.py` | ASR 单模块测试：wav 流式喂帧 → 拼句 → CER |
| `tests/test_translator.py` | Translator 单模块测试：文本输入 → 关键词 + yield 单调性 |
| `tests/test_tts.py` | TTS 单模块测试：合成 → sanity + ASR round-trip CER |
| `tests/fixtures/translations_ref.json` | Translator 参考（输入文本 + 关键词组） |
| `tests/fixtures/hello_singapore_zh.wav` | ✅ 已存在 |
| `tests/fixtures/hello_singapore_zh.txt` | ✅ 已存在 |

不修改任何 `models/`、`server.py`、`pipeline.py`、`client.py`、`config.py`。

---

## 文件详细设计

### 1. `tests/_utils.py`（共享工具）

导出：

```python
import numpy as np
import soundfile as sf
from difflib import SequenceMatcher

SAMPLE_RATE = 16000
FRAME_SIZE  = 512  # 32ms @16k，与 config.py 一致

_PUNCT = set("，。！？,.!?;；:：、 \t\n\r")

def load_wav(path: str) -> tuple[np.ndarray, int]:
    """读 wav → float32 mono @ 原 sr"""

def save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """float32 → PCM16 wav（用 soundfile 避免 torchcodec 依赖）"""

def normalize_text(s: str) -> str:
    """去空白 + 去中英文常见标点，大小写不变（CJK 无意义）"""

def cer(ref: str, hyp: str) -> float:
    """字符级错误率：1 - SequenceMatcher.ratio()，先做 normalize"""

def iter_frames(audio: np.ndarray, frame_size: int = FRAME_SIZE):
    """yield 定长帧；不足 frame_size 的尾巴补零到整帧"""

class TimestampLog:
    """记录 (label, wall_time_sec) 事件，支持 pretty print 和相对时间"""
    def mark(self, label: str): ...
    def dump(self) -> str: ...
```

关键点：
- `save_wav` 用 `soundfile.write(..., subtype="PCM_16")`，**不要** `torchaudio.save`（env 没装 torchcodec，已在 gen_test_audio.py 踩过）。
- `iter_frames` 给 ASR 测试把 wav 切成 `FRAME_SIZE=512` 的块。
- `cer` 实现基于 `difflib.SequenceMatcher`，够用且零依赖。

---

### 2. `tests/test_asr.py`

**输入**：`tests/fixtures/hello_singapore_zh.wav` + `.txt`
**预期**：识别文本对参考文本 `CER < 0.15`，且至少产出一段文字（来自 `feed_audio` 返回 或 `finish_session` 尾部）。

流程：

```
1. audio, sr = load_wav(WAV_PATH); assert sr == 16000
2. ref_text = open(TXT_PATH).read().strip()
3. asr = StreamingASR()
4. asr.start_session()
5. collected: list[str] = []
6. for frame in iter_frames(audio, 512):
       t0 = time.time()
       result = asr.feed_audio(frame)
       tlog.mark(f"feed_audio rt={time.time()-t0:.3f}s")
       if result:
           collected.append(result)
           tlog.mark(f"SENTENCE_EMIT: {result!r}")
7. tail = asr.finish_session()
   if tail: collected.append(tail)
8. hyp_text = "".join(collected)
9. err = cer(ref_text, hyp_text)
10. print 诊断：ref/hyp/err/backend/timing
11. assert err < 0.15, 否则 sys.exit(1)
```

失败时输出：
- `ref`、`hyp`
- `err`（CER 数值）
- `collected` 列表（看是不是空，或分多段）
- `DEVICE` 值、`len(audio)/sr` 时长、全部 feed_audio 调用的 wall-clock 时间
- 是否走的 Transformers 批量分支（通过检查 `asr.model` 或 `hasattr` 分支变量）

注意：**不** `sleep(32ms)` 模拟实时，直接连续喂帧。理由：当前测试只校验正确性；macOS batch 模式下喂完所有帧才会触发第二次 batch；加 sleep 只会让测试慢且不测任何新东西。

---

### 3. `tests/test_translator.py`

**输入**：从 `tests/fixtures/translations_ref.json` 读：

```json
{
  "cases": [
    {
      "name": "zh_to_en_hello_singapore",
      "text": "你好，我正在新加坡与你通话。",
      "src": "zh",
      "tgt": "en",
      "keyword_groups": [
        ["hello", "hi", "hey", "greetings"],
        ["singapore"],
        ["call", "calling", "speak", "speaking", "talk", "talking", "phone", "on the phone"]
      ]
    }
  ]
}
```

**预期**：
- `translate()` 一次性模式返回非空，不含 `<think>`，每个关键词组至少命中 1 个（大小写无关）。
- `translate_streaming()` 至少 yield 2 次，且 `len(yield[i]) >= len(yield[i-1])`，最后一次 yield 同样通过关键词组检查。

流程：

```
1. tr = StreamingTranslator()
2. for case in cases:
    a. final = tr.translate(case.text, case.src, case.tgt)
       assert "<think>" not in final
       assert _all_groups_hit(final, case.keyword_groups)
    b. yields = []
       ts_first = None
       for partial in tr.translate_streaming(case.text, case.src, case.tgt):
           yields.append((time.time(), partial))
           if ts_first is None: ts_first = time.time()
       assert len(yields) >= 2
       lengths = [len(p) for _, p in yields]
       assert lengths == sorted(lengths)    # 单调非递减
       assert lengths[-1] > lengths[0]       # 确实增长
       last_text = yields[-1][1]
       assert "<think>" not in last_text
       assert _all_groups_hit(last_text, case.keyword_groups)
3. 全部通过 print 汇总并 exit(0)
```

`_all_groups_hit(text, groups)`：`text_lower = text.lower()`；对每个 group 要求 `any(kw in text_lower for kw in group)`。

失败时打印：
- case 名
- 原文、final / last streaming 输出
- 哪一组没命中
- yield 序列长度与内容片段（前 80 char）

---

### 4. `tests/test_tts.py`

**输入**：参考文本 `"你好，我正在新加坡与你通话。"`，`language="Chinese"`
**预期**：
- sanity：`audio.ndim == 1`（或 `(1, T)`，测试里 `.squeeze()`），`dtype` 可转 float32，非空，`1.0s ≤ duration ≤ 10.0s`，无 NaN/Inf，`sr > 0`。
- round-trip：把合成的音频重采样到 16kHz，过 `StreamingASR`，识别结果 `CER(ref, hyp) < 0.25`（比 ASR 单测宽松，因为 TTS→ASR 串联误差会叠加）。

流程：

```
1. tts = TTSEngine()
2. audio, sr = tts.synthesize(REF_TEXT, "Chinese")
3. if isinstance(audio, torch.Tensor): audio = audio.detach().cpu().float().numpy()
   audio = np.squeeze(audio)
4. assert audio.ndim == 1 and audio.size > 0
   assert not np.isnan(audio).any() and not np.isinf(audio).any()
   dur = audio.size / sr
   assert 1.0 <= dur <= 10.0
5. # round-trip
   if sr != 16000:
       wav16 = torchaudio.functional.resample(torch.from_numpy(audio), sr, 16000).numpy()
   else:
       wav16 = audio
   asr = StreamingASR()
   asr.start_session()
   collected = []
   for frame in iter_frames(wav16, 512):
       r = asr.feed_audio(frame)
       if r: collected.append(r)
   tail = asr.finish_session()
   if tail: collected.append(tail)
   hyp = "".join(collected)
6. err = cer(REF_TEXT, hyp)
7. print 诊断；assert err < 0.25
```

注意事项：
- TTS 模型和 ASR 模型都要加载 → 这个脚本最慢，放最后跑。
- 如果 round-trip CER 卡在 0.20–0.25 附近，优先怀疑 ASR 而非 TTS（ASR 测试在独立文件里已用同一段音频做过校验，可交叉诊断）。
- round-trip 的 CER 阈值 0.25 是经验值，允许实际执行时根据首次结果放宽/收紧。

---

## 执行方式

单个测试（失败会非零退出，便于我循环调试）：

```bash
/opt/homebrew/Caskroom/miniconda/base/envs/yisounda/bin/python tests/test_asr.py
/opt/homebrew/Caskroom/miniconda/base/envs/yisounda/bin/python tests/test_translator.py
/opt/homebrew/Caskroom/miniconda/base/envs/yisounda/bin/python tests/test_tts.py
```

顺序跑全部：

```bash
PY=/opt/homebrew/Caskroom/miniconda/base/envs/yisounda/bin/python
cd /Users/bytedance/0-Code/YiSounda
$PY tests/test_asr.py && $PY tests/test_translator.py && $PY tests/test_tts.py
```

每个脚本的输出格式统一：

```
[test_asr] DEVICE=mps  audio=3.12s  backend=transformers
[test_asr] REF: 你好，我正在新加坡与你通话。
[test_asr] HYP: ...
[test_asr] CER: 0.071
[test_asr] RESULT: PASS
```

失败时 `RESULT: FAIL`，附加异常 trace 与具体不达标指标。

---

## 自动调试循环（给我自己用）

1. 跑 `tests/test_asr.py`
2. 失败 → 读取 stdout/stderr → 定位问题：模型加载失败 / 音频格式错 / ASR 分支走错 / CER 过高
3. 只修 `models/asr.py` 或 `config.py`（不改测试，除非阈值明显不合理）
4. 再跑，直到 PASS
5. 进入 `tests/test_translator.py`，重复
6. 最后 `tests/test_tts.py`，重复
7. 三个全绿后汇报给用户，由用户决定下一步（启动 server+client 真实链路验证）

---

## 验证

三个脚本全部 `RESULT: PASS` 即视为模块级任务完成。之后手工验证端到端：

```bash
$PY server.py &
$PY client_2.py --input tests/fixtures/hello_singapore_zh.wav --inp-lang zh --opt-lang en
```

（client_2.py 支持 `--input` 读 wav 模拟麦克风，见 `client_2.py:148`，不需要为端到端再造模拟器。）

---

## 不在本 plan 范围内

- Pipeline E2E 测试（用户明确不做）
- server / client 的 WebSocket 协议测试
- 真实麦克风 / 真实耳机的硬件联调
- 模型精度调参（system prompt、温度、beam search 等）
- 对 `models/` 下任何文件的修改（除非测试发现 bug 需要修复）
