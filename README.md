<div align="center">
<h1>
FireRedASR2S
<br>
A SOTA Industrial-Grade All-in-One ASR System
</h1>

</div>

[[Paper]](https://arxiv.org/pdf/2501.14350)
[[Model]](https://huggingface.co/FireRedTeam)
[[Blog]](https://fireredteam.github.io/demos/firered_asr/)
[[Demo]](https://huggingface.co/spaces/FireRedTeam/FireRedASR)


FireRedASR2S is a state-of-the-art (SOTA), industrial-grade, all-in-one ASR system with ASR, VAD, LID, and Punc modules. All modules achieve SOTA performance:
- **FireRedASR2**: Automatic Speech Recognition (ASR) supporting Chinese (Mandarin, 20+ dialects/accents), English, code-switching, and singing lyrics recognition. 2.89% average CER on Mandarin (4 test sets), 11.55% on Chinese dialects (19 test sets), outperforming Doubao-ASR, Qwen3-ASR-1.7B, Fun-ASR, and Fun-ASR-Nano-2512. FireRedASR2-AED also supports word-level timestamps and confidence scores.
- **FireRedVAD**: Voice Activity Detection (VAD) supporting speech/singing/music in 100+ languages. 97.57% F1, outperforming Silero-VAD, TEN-VAD, and FunASR-VAD. Supports non-streaming/streaming VAD and Audio Event Detection.
- **FireRedLID**: Spoken Language Identification (LID) supporting 100+ languages and 20+ Chinese dialects/accents. 97.18% accuracy, outperforming Whisper and SpeechBrain-LID.
- **FireRedPunc**: Punctuation Prediction (Punc) for Chinese and English. 78.90% average F1, outperforming FunASR-Punc (62.77%).

*`2S`: `2`nd-generation FireRedASR, now expanded to an all-in-one ASR `S`ystem*


## 🔥 News
- [2026.02.13] 🚀 Support TensorRT-LLM inference acceleration for FireRedASR2-AED (contributed by NVIDIA). Benchmark on AISHELL-1 test set shows **12.7x speedup** over PyTorch baseline (single H20).
- [2026.02.12] We release FireRedASR2S (FireRedASR2-AED, FireRedVAD, FireRedLID, and FireRedPunc) with **model weights and inference code**. Download links below. Technical report and finetuning code coming soon.



## Available Models and Languages

|Model|Supported Languages & Dialects|Download|
|:-------------:|:---------------------------------:|:----------:|
|FireRedASR2| Chinese (Mandarin and 20+ dialects/accents<sup>*</sup>), English, Code-Switching | [🤗](https://huggingface.co/FireRedTeam/FireRedASR2-AED) \| [🤖](https://www.modelscope.cn/models/xukaituo/FireRedASR2-AED/)|
|FireRedVAD | 100+ languages, 20+ Chinese dialects/accents<sup>*</sup> | [🤗](https://huggingface.co/FireRedTeam/FireRedVAD) \| [🤖](https://www.modelscope.cn/models/xukaituo/FireRedVAD/)|
|FireRedLID | 100+ languages, 20+ Chinese dialects/accents<sup>*</sup> | [🤗](https://huggingface.co/FireRedTeam/FireRedLID) \| [🤖](https://www.modelscope.cn/models/xukaituo/FireRedLID/)|
|FireRedPunc| Chinese, English | [🤗](https://huggingface.co/FireRedTeam/FireRedPunc) \| [🤖](https://www.modelscope.cn/models/xukaituo/FireRedPunc/)|

<sup>*</sup>Supported Chinese dialects/accents: Cantonese (Hong Kong & Guangdong), Sichuan, Shanghai, Wu, Minnan, Anhui, Fujian, Gansu, Guizhou, Hebei, Henan, Hubei, Hunan, Jiangxi, Liaoning, Ningxia, Shaanxi, Shanxi, Shandong, Tianjin, Yunnan, etc.



## Method
### FireRedASR2
FireRedASR2 builds upon [FireRedASR](https://github.com/FireRedTeam/FireRedASR) with improved accuracy, designed to meet diverse requirements in superior performance and optimal efficiency across various applications. It comprises two variants:
- **FireRedASR2-LLM**: Designed to achieve state-of-the-art performance and to enable seamless end-to-end speech interaction. It adopts an Encoder-Adapter-LLM framework leveraging large language model (LLM) capabilities.
- **FireRedASR2-AED**: Designed to balance high performance and computational efficiency and to serve as an effective speech representation module in LLM-based speech models. It utilizes an Attention-based Encoder-Decoder (AED) architecture.

![Model](./assets/FireRedASR2_model.png)

### Other Modules
- **FireRedVAD**: DFSMN-based non-streaming/streaming Voice Activity Detection and Audio Event Detection.
- **FireRedLID**: FireRedASR2-based Spoken Language Identification. See [FireRedLID README](./fireredasr2s/fireredlid/README.md) for language details.
- **FireRedPunc**: BERT-based Punctuation Prediction.


## Evaluation
### FireRedASR2
Metrics: Character Error Rate (CER%) for Chinese and Word Error Rate (WER%) for English. Lower is better.

We evaluate FireRedASR2 on 24 public test sets covering Mandarin, 20+ Chinese dialects/accents, and singing.

- **Mandarin (4 test sets)**: 2.89% (LLM) / 3.05% (AED) average CER, outperforming Doubao-ASR (3.69%), Qwen3-ASR-1.7B (3.76%), Fun-ASR (4.16%) and Fun-ASR-Nano-2512 (4.55%).
- **Dialects (19 test sets)**: 11.55% (LLM) / 11.67% (AED) average CER, outperforming Doubao-ASR (15.39%), Qwen3-ASR-1.7B (11.85%), Fun-ASR (12.76%) and Fun-ASR-Nano-2512 (15.07%).

*Note: ws=WenetSpeech, md=MagicData, conv=Conversational, daily=Daily-use.*

|ID|Testset\Model|FireRedASR2-LLM|FireRedASR2-AED|Doubao-ASR|Qwen3-ASR|Fun-ASR|Fun-ASR-Nano|
|:--:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  |**Average CER<br>(All, 1-24)**   |**9.67** |**9.80** |12.98 |10.12 |10.92 |12.81 |
|  |**Average CER<br>(Mandarin, 1-4)**   |**2.89** |**3.05** |3.69 |3.76 |4.16 |4.55 |
|  |**Average CER<br>(Dialects, 5-23)**  |**11.55**|**11.67**|15.39|11.85|12.76|15.07|
|1 |aishell1          |0.64 |0.57 |1.52 |1.48 |1.64 |1.96 |
|2 |aishell2          |2.15 |2.51 |2.77 |2.71 |2.38 |3.02 |
|3 |ws-net            |4.44 |4.57 |5.73 |4.97 |6.85 |6.93 |
|4 |ws-meeting        |4.32 |4.53 |4.74 |5.88 |5.78 |6.29 |
|5 |kespeech          |3.08 |3.60 |5.38 |5.10 |5.36 |7.66 |
|6 |ws-yue-short      |5.14 |5.15 |10.51|5.82 |7.34 |8.82 |
|7 |ws-yue-long       |8.71 |8.54 |11.39|8.85 |10.14|11.36|
|8 |ws-chuan-easy     |10.90|10.60|11.33|11.99|12.46|14.05|
|9 |ws-chuan-hard     |20.71|21.35|20.77|21.63|22.49|25.32|
|10|md-heavy          |7.42 |7.43 |7.69 |8.02 |9.13 |9.97 |
|11|md-yue-conv       |12.23|11.66|26.25|9.76 |33.71|15.68|
|12|md-yue-daily      |3.61 |3.35 |12.82|3.66 |2.69 |5.67 |
|13|md-yue-vehicle    |4.50 |4.83 |8.66 |4.28 |6.00 |7.04 |
|14|md-chuan-conv     |13.18|13.07|11.77|14.35|14.01|17.11|
|15|md-chuan-daily    |4.90 |5.17 |3.90 |4.93 |3.98 |5.95 |
|16|md-shanghai-conv  |28.70|27.02|45.15|29.77|25.49|37.08|
|17|md-shanghai-daily |24.94|24.18|44.06|23.93|12.55|28.77|
|18|md-wu             |7.15 |7.14 |7.70 |7.57 |10.63|10.56|
|19|md-zhengzhou-conv |10.20|10.65|9.83 |9.55 |10.85|13.09|
|20|md-zhengzhou-daily|5.80 |6.26 |5.77 |5.88 |6.29 |8.18 |
|21|md-wuhan          |9.60 |10.81|9.94 |10.22|4.34 |8.70 |
|22|md-tianjin        |15.45|15.30|15.79|16.16|19.27|22.03|
|23|md-changsha       |23.18|25.64|23.76|23.70|25.66|29.23|
|24|opencpop          |1.12 |1.17 |4.36 |2.57 |3.05 |2.95 |

Doubao-ASR (volc.seedasr.auc) tested in early February 2026, and Fun-ASR tested in late November 2025. Our ASR training data does not include any Chinese dialect or accented speech data from MagicData.
- Doubao-ASR (API): https://www.volcengine.com/docs/6561/1354868
- Qwen3-ASR (1.7B): https://github.com/QwenLM/Qwen3-ASR
- Fun-ASR (API): https://help.aliyun.com/zh/model-studio/recording-file-recognition
- Fun-ASR-Nano-2512: https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512


### FireRedVAD
We evaluate FireRedVAD on FLEURS-VAD-102, a multilingual VAD benchmark covering 102 languages.

FireRedVAD achieves SOTA performance, outperforming Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD.

|Metric\Model|FireRedVAD|[Silero-VAD](https://github.com/snakers4/silero-vad)|[TEN-VAD](https://github.com/TEN-framework/ten-vad)|[FunASR-VAD](https://modelscope.cn/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch)|[WebRTC-VAD](https://github.com/wiseman/py-webrtcvad)|
|:-------:|:-----:|:------:|:------:|:------:|:------:|
|AUC-ROC↑  |**99.60**|97.99|97.81|-    |-    |
|F1 score↑ |**97.57**|95.95|95.19|90.91|52.30|
|False Alarm Rate↓  |**2.69** |9.41 |15.47|44.03|2.83 |
|Miss Rate↓|3.62     |3.95 |2.95 |0.42 |64.15|

<sup>*</sup>FLEURS-VAD-102: We randomly selected ~100 audio files per language from [FLEURS test set](https://huggingface.co/datasets/google/fleurs), resulting in 9,443 audio files with manually annotated binary VAD labels (speech=1, silence=0). This VAD testset will be open sourced (coming soon).

Note: FunASR-VAD achieves low Miss Rate but at the cost of high False Alarm Rate (44.03%), indicating over-prediction of speech segments.


### FireRedLID
Metric: Utterance-level LID Accuracy (%). Higher is better.

We evaluate FireRedLID on multilingual and Chinese dialect benchmarks.

FireRedLID achieves SOTA performance, outperforming Whisper, SpeechBrain-LID, and Dolphin.

|Testset\Model|Languages|FireRedLID|[Whisper](https://github.com/openai/whisper)|[SpeechBrain](https://huggingface.co/speechbrain/lang-id-voxlingua107-ecapa)|[Dolphin](https://github.com/DataoceanAI/Dolphin)|
|:-----------------:|:---------:|:---------:|:-----:|:---------:|:-----:|
|FLEURS test         |82 languages |**97.18**     |79.41  |92.91      |-|
|CommonVoice test    |74 languages |**92.07**    |80.81  |78.75      |-|
|KeSpeech + MagicData|20+ Chinese dialects/accents |**88.47**     |-|-|69.01|


### FireRedPunc
Metric: Precision/Recall/F1 Score (%). Higher is better.

We evaluate FireRedPunc on multi-domain Chinese and English benchmarks.

FireRedPunc achieves SOTA performance, outperforming FunASR-Punc (CT-Transformer).

|Testset\Model|#Sentences|FireRedPunc|[FunASR-Punc](https://www.modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch)|
|:------------------:|:---------:|:--------------:|:-----------------:|
|Multi-domain Chinese| 88,644    |**82.84 / 83.08 / 82.96** | 77.27 / 74.03 / 75.62 |
|Multi-domain English| 28,641    |**78.40 / 71.57 / 74.83** | 55.79 / 45.15 / 49.91 |
|Average F1 Score    | -         |**78.90** | 62.77 |




## Quick Start
### Setup
1. Create a clean Python environment:
```bash
$ conda create --name fireredasr2s python=3.10
$ conda activate fireredasr2s
$ git clone https://github.com/FireRedTeam/FireRedASR2S.git
$ cd FireRedASR2S  # or fireredasr2s
```

2. Install dependencies and set up PATH and PYTHONPATH:
```bash
$ pip install -r requirements.txt
$ export PATH=$PWD/fireredasr2s/:$PATH
$ export PYTHONPATH=$PWD/:$PYTHONPATH
```

3. Download models:
```bash
# Download via ModelScope (recommended for users in China)
pip install -U modelscope
modelscope download --model xukaituo/FireRedASR2-AED --local_dir ./pretrained_models/FireRedASR2-AED
modelscope download --model xukaituo/FireRedVAD --local_dir ./pretrained_models/FireRedVAD
modelscope download --model xukaituo/FireRedLID --local_dir ./pretrained_models/FireRedLID
modelscope download --model xukaituo/FireRedPunc --local_dir ./pretrained_models/FireRedPunc

# Download via Hugging Face
pip install -U "huggingface_hub[cli]"
huggingface-cli download FireRedTeam/FireRedASR2-AED --local-dir ./pretrained_models/FireRedASR2-AED
huggingface-cli download FireRedTeam/FireRedVAD --local-dir ./pretrained_models/FireRedVAD
huggingface-cli download FireRedTeam/FireRedLID --local-dir ./pretrained_models/FireRedLID
huggingface-cli download FireRedTeam/FireRedPunc --local-dir ./pretrained_models/FireRedPunc
```

4. Convert your audio to **16kHz 16-bit mono PCM** format if needed:
```bash
$ ffmpeg -i <input_audio_path> -ar 16000 -ac 1 -acodec pcm_s16le -f wav <output_wav_path>
```

### Script Usage
```bash
$ cd examples_infer/asr_system
$ bash inference_asr_system.sh
```

### Command-line Usage
```bash
$ fireredasr2s-cli --help
$ fireredasr2s-cli --wav_paths "assets/hello_zh.wav" "assets/hello_en.wav" --outdir output
$ cat output/result.jsonl 
# {"uttid": "hello_zh", "text": "你好世界。", "sentences": [{"start_ms": 310, "end_ms": 1840, "text": "你好世界。", "asr_confidence": 0.875, "lang": "zh mandarin", "lang_confidence": 0.999}], "vad_segments_ms": [[310, 1840]], "dur_s": 2.32, "words": [{"start_ms": 490, "end_ms": 690, "text": "你"}, {"start_ms": 690, "end_ms": 1090, "text": "好"}, {"start_ms": 1090, "end_ms": 1330, "text": "世"}, {"start_ms": 1330, "end_ms": 1795, "text": "界"}], "wav_path": "assets/hello_zh.wav"}
# {"uttid": "hello_en", "text": "Hello speech.", "sentences": [{"start_ms": 120, "end_ms": 1840, "text": "Hello speech.", "asr_confidence": 0.833, "lang": "en", "lang_confidence": 0.998}], "vad_segments_ms": [[120, 1840]], "dur_s": 2.24, "words": [{"start_ms": 340, "end_ms": 1020, "text": "hello"}, {"start_ms": 1020, "end_ms": 1666, "text": "speech"}], "wav_path": "assets/hello_en.wav"}
```

### Python API Usage
```python
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

asr_system_config = FireRedAsr2SystemConfig()  # Use default config
asr_system = FireRedAsr2System(asr_system_config)

result = asr_system.process("assets/hello_zh.wav")
print(result)
# {'uttid': 'tmpid', 'text': '你好世界。', 'sentences': [{'start_ms': 440, 'end_ms': 1820, 'text': '你好世界。', 'asr_confidence': 0.868, 'lang': 'zh mandarin', 'lang_confidence': 0.999}], 'vad_segments_ms': [(440, 1820)], 'dur_s': 2.32, 'words': [], 'wav_path': 'assets/hello_zh.wav'}

result = asr_system.process("assets/hello_en.wav")
print(result)
# {'uttid': 'tmpid', 'text': 'Hello speech.', 'sentences': [{'start_ms': 260, 'end_ms': 1820, 'text': 'Hello speech.', 'asr_confidence': 0.933, 'lang': 'en', 'lang_confidence': 0.993}], 'vad_segments_ms': [(260, 1820)], 'dur_s': 2.24, 'words': [], 'wav_path': 'assets/hello_en.wav'}
```



## Usage of Each Module
The four components under `fireredasr2s`, i.e. `fireredasr2`, `fireredvad`, `fireredlid`, and `fireredpunc` are self-contained and designed to work as a standalone modules. You can use any of them independently without depending on the others. `FireRedVAD` and `FireRedLID` will also be open-sourced as standalone libraries in separate repositories.

### Script Usage
```bash
# ASR
$ cd examples_infer/asr
$ bash inference_asr_aed.sh
$ bash inference_asr_llm.sh

# VAD & AED (Audio Event Detection)
$ cd examples_infer/vad
$ bash inference_vad.sh
$ bash inference_streamvad.sh
$ bash inference_aed.sh

# LID
$ cd examples_infer/lid
$ bash inference_lid.sh

# Punc
$ cd examples_infer/punc
$ bash inference_punc.sh
```


### Python API Usage
Set up `PYTHONPATH` first: `export PYTHONPATH=$PWD/:$PYTHONPATH`

#### ASR
```python
from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config

batch_uttid = ["hello_zh", "hello_en"]
batch_wav_path = ["assets/hello_zh.wav", "assets/hello_en.wav"]

# FireRedASR2-AED
asr_config = FireRedAsr2Config(
    use_gpu=True,
    use_half=False,
    beam_size=3,
    nbest=1,
    decode_max_len=0,
    softmax_smoothing=1.25,
    aed_length_penalty=0.6,
    eos_penalty=1.0,
    return_timestamp=True
)
model = FireRedAsr2.from_pretrained("aed", "pretrained_models/FireRedASR2-AED", asr_config)
results = model.transcribe(batch_uttid, batch_wav_path)
print(results)
# [{'uttid': 'hello_zh', 'text': '你好世界', 'confidence': 0.971, 'dur_s': 2.32, 'rtf': '0.0870', 'wav': 'assets/hello_zh.wav', 'timestamp': [('你', 0.42, 0.66), ('好', 0.66, 1.1), ('世', 1.1, 1.34), ('界', 1.34, 2.039)]}, {'uttid': 'hello_en', 'text': 'hello speech', 'confidence': 0.943, 'dur_s': 2.24, 'rtf': '0.0870', 'wav': 'assets/hello_en.wav', 'timestamp': [('hello', 0.34, 0.98), ('speech', 0.98, 1.766)]}]

# FireRedASR2-LLM
asr_config = FireRedAsr2Config(
    use_gpu=True,
    decode_min_len=0,
    repetition_penalty=1.0,
    llm_length_penalty=0.0,
    temperature=1.0
)
model = FireRedAsr2.from_pretrained("llm", "pretrained_models/FireRedASR2-LLM", asr_config)
results = model.transcribe(batch_uttid, batch_wav_path)
print(results)
# [{'uttid': 'hello_zh', 'text': '你好世界', 'rtf': '0.0681', 'wav': 'assets/hello_zh.wav'}, {'uttid': 'hello_en', 'text': 'hello speech', 'rtf': '0.0681', 'wav': 'assets/hello_en.wav'}]
```


#### VAD
```python
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

vad_config = FireRedVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000)
vad = FireRedVad.from_pretrained("pretrained_models/FireRedVAD/VAD", vad_config)

result, probs = vad.detect("assets/hello_zh.wav")

print(result)
# {'dur': 2.32, 'timestamps': [(0.44, 1.82)], 'wav_path': 'assets/hello_zh.wav'}
```


#### Stream VAD
<details>
<summary>Click to expand</summary>

```python
from fireredasr2s.fireredvad import FireRedStreamVad, FireRedStreamVadConfig

vad_config=FireRedStreamVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    pad_start_frame=5,
    min_speech_frame=8,
    max_speech_frame=2000,
    min_silence_frame=20,
    chunk_max_frame=30000)
stream_vad = FireRedStreamVad.from_pretrained("pretrained_models/FireRedVAD/Stream-VAD", vad_config)

frame_results, result = stream_vad.detect_full("assets/hello_zh.wav")

print(result)
# {'dur': 2.32, 'timestamps': [(0.46, 1.84)], 'wav_path': 'assets/hello_zh.wav'}
```
</details>


#### Audio Event Detection (AED)
<details>
<summary>Click to expand</summary>

```python
from fireredasr2s.fireredvad import FireRedAed, FireRedAedConfig

aed_config=FireRedAedConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    singing_threshold=0.5,
    music_threshold=0.5,
    min_event_frame=20,
    max_event_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000)
aed = FireRedAed.from_pretrained("pretrained_models/FireRedVAD/AED", aed_config)

result, probs = aed.detect("assets/event.wav")

print(result)
# {'dur': 22.016, 'event2timestamps': {'speech': [(0.4, 3.56), (3.66, 9.08), (9.27, 9.77), (10.78, 21.76)], 'singing': [(1.79, 19.96), (19.97, 22.016)], 'music': [(0.09, 12.32), (12.33, 22.016)]}, 'event2ratio': {'speech': 0.848, 'singing': 0.905, 'music': 0.991}, 'wav_path': 'assets/event.wav'}
```
</details>


#### LID
<details>
<summary>Click to expand</summary>

```python
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig

batch_uttid = ["hello_zh", "hello_en"]
batch_wav_path = ["assets/hello_zh.wav", "assets/hello_en.wav"]

config = FireRedLidConfig(use_gpu=True, use_half=False)
model = FireRedLid.from_pretrained("pretrained_models/FireRedLID", config)

results = model.process(batch_uttid, batch_wav_path)
print(results)
# [{'uttid': 'hello_zh', 'lang': 'zh mandarin', 'confidence': 0.996, 'dur_s': 2.32, 'rtf': '0.0741', 'wav': 'assets/hello_zh.wav'}, {'uttid': 'hello_en', 'lang': 'en', 'confidence': 0.996, 'dur_s': 2.24, 'rtf': '0.0741', 'wav': 'assets/hello_en.wav'}]
```
</details>


#### Punc
<details>
<summary>Click to expand</summary>

```python
from fireredasr2s.fireredpunc.punc import FireRedPunc, FireRedPuncConfig

config = FireRedPuncConfig(use_gpu=True)
model = FireRedPunc.from_pretrained("pretrained_models/FireRedPunc", config)

batch_text = ["你好世界", "Hello world"]
results = model.process(batch_text)

print(results)
# [{'punc_text': '你好世界。', 'origin_text': '你好世界'}, {'punc_text': 'Hello world!', 'origin_text': 'Hello world'}]
```
</details>


#### ASR System
```python
from fireredasr2s.fireredasr2 import FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVadConfig
from fireredasr2s import FireRedAsr2System, FireRedAsr2SystemConfig

vad_config = FireRedVadConfig(
    use_gpu=False,
    smooth_window_size=5,
    speech_threshold=0.4,
    min_speech_frame=20,
    max_speech_frame=2000,
    min_silence_frame=20,
    merge_silence_frame=0,
    extend_speech_frame=0,
    chunk_max_frame=30000
)
lid_config = FireRedLidConfig(use_gpu=True, use_half=False)
asr_config = FireRedAsr2Config(
    use_gpu=True,
    use_half=False,
    beam_size=3,
    nbest=1,
    decode_max_len=0,
    softmax_smoothing=1.25,
    aed_length_penalty=0.6,
    eos_penalty=1.0,
    return_timestamp=True
)
punc_config = FireRedPuncConfig(use_gpu=True)

asr_system_config = FireRedAsr2SystemConfig(
    "pretrained_models/FireRedVAD/VAD",
    "pretrained_models/FireRedLID",
    "aed", "pretrained_models/FireRedASR2-AED",
    "pretrained_models/FireRedPunc",
    vad_config, lid_config, asr_config, punc_config,
    enable_vad=1, enable_lid=1, enable_punc=1
)
asr_system = FireRedAsr2System(asr_system_config)

batch_uttid = ["hello_zh", "hello_en"]
batch_wav_path = ["assets/hello_zh.wav", "assets/hello_en.wav"]
for wav_path, uttid in zip(batch_wav_path, batch_uttid):
    result = asr_system.process(wav_path, uttid)
    print(result)
# {'uttid': 'hello_zh', 'text': '你好世界。', 'sentences': [{'start_ms': 440, 'end_ms': 1820, 'text': '你好世界。', 'asr_confidence': 0.868, 'lang': 'zh mandarin', 'lang_confidence': 0.999}], 'vad_segments_ms': [(440, 1820)], 'dur_s': 2.32, 'words': [{'start_ms': 540, 'end_ms': 700, 'text': '你'}, {'start_ms': 700, 'end_ms': 1100, 'text': '好'}, {'start_ms': 1100, 'end_ms': 1300, 'text': '世'}, {'start_ms': 1300, 'end_ms': 1765, 'text': '界'}], 'wav_path': 'assets/hello_zh.wav'}
# {'uttid': 'hello_en', 'text': 'Hello speech.', 'sentences': [{'start_ms': 260, 'end_ms': 1820, 'text': 'Hello speech.', 'asr_confidence': 0.933, 'lang': 'en', 'lang_confidence': 0.993}], 'vad_segments_ms': [(260, 1820)], 'dur_s': 2.24, 'words': [{'start_ms': 400, 'end_ms': 960, 'text': 'hello'}, {'start_ms': 960, 'end_ms': 1666, 'text': 'speech'}], 'wav_path': 'assets/hello_en.wav'}
```



## FAQ
**Q: What audio format is supported?**

16kHz 16-bit mono PCM wav. Use ffmpeg to convert other formats: `ffmpeg -i <input_audio_path> -ar 16000 -ac 1 -acodec pcm_s16le -f wav <output_wav_path>`

**Q: What are the input length limitations of ASR models?**

- FireRedASR2-AED supports audio input up to 60s. Input longer than 60s may cause hallucination issues, and input exceeding 200s will trigger positional encoding errors.
- FireRedASR2-LLM supports audio input up to 30s. The behavior for longer input is untested.


## Acknowledgements
Thanks to the following open-source works:
- [Qwen](https://huggingface.co/Qwen)
- [WenetSpeech-Yue](https://github.com/ASLP-lab/WenetSpeech-Yue)
- [WenetSpeech-Chuan](https://github.com/ASLP-lab/WenetSpeech-Chuan)
