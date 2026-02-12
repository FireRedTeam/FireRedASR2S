# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Kai Huang, Yan Jia, Junjie Chen, Wenpeng Li)

import logging
import re
from dataclasses import dataclass, field

import soundfile as sf

from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
from fireredasr2s.fireredlid import FireRedLid, FireRedLidConfig
from fireredasr2s.fireredpunc import FireRedPunc, FireRedPuncConfig
from fireredasr2s.fireredvad import FireRedVad, FireRedVadConfig

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
logger = logging.getLogger("fireredasr2s.asr_system")


@dataclass
class FireRedAsr2SystemConfig:
    vad_model_dir: str = "pretrained_models/FireRedVAD/VAD"
    lid_model_dir: str = "pretrained_models/FireRedLID"
    asr_type: str = "aed"
    asr_model_dir: str = "pretrained_models/FireRedASR2-AED"
    punc_model_dir: str = "pretrained_models/FireRedPunc"
    vad_config: FireRedVadConfig = field(default_factory=FireRedVadConfig)
    lid_config: FireRedLidConfig = field(default_factory=FireRedLidConfig)
    asr_config: FireRedAsr2Config = field(default_factory=FireRedAsr2Config)
    punc_config: FireRedPuncConfig = field(default_factory=FireRedPuncConfig)
    asr_batch_size: int = 1
    punc_batch_size: int = 1
    enable_vad: bool = True
    enable_lid: bool = True
    enable_punc: bool = True


class FireRedAsr2System:
    def __init__(self, config):
        c = config
        self.vad = FireRedVad.from_pretrained(c.vad_model_dir, c.vad_config) if c.enable_vad else None
        self.lid = FireRedLid.from_pretrained(c.lid_model_dir, c.lid_config) if c.enable_lid else None
        self.asr = FireRedAsr2.from_pretrained(c.asr_type, c.asr_model_dir, c.asr_config)
        self.punc = FireRedPunc.from_pretrained(c.punc_model_dir, c.punc_config) if c.enable_punc else None
        self.config = config

    def process(self, wav_path, uttid="tmpid"):
        wav_np, sample_rate = sf.read(wav_path, dtype="int16")
        dur = wav_np.shape[0]/sample_rate

        # 1. VAD
        if self.config.enable_vad:
            vad_result, prob = self.vad.detect(wav_path)
            vad_segments = vad_result["timestamps"]
            logger.info(f"VAD: {vad_result}")
        else:
            vad_segments = [(0, dur)]
            vad_result = {"timestamps" : vad_segments}

        # 2. VAD output to ASR input
        asr_results = []
        lid_results = []
        assert sample_rate == 16000
        batch_asr_uttid = []
        batch_asr_wav = []
        for j, (start_s, end_s) in enumerate(vad_segments):
            wav_segment = wav_np[int(start_s*sample_rate):int(end_s*sample_rate)]
            vad_uttid = f"{uttid}_s{int(start_s*1000)}_e{int(end_s*1000)}"
            batch_asr_uttid.append(vad_uttid)
            batch_asr_wav.append((sample_rate, wav_segment))
            if len(batch_asr_uttid) < self.config.asr_batch_size and j != len(vad_segments) - 1:
                continue

            # 3. ASR
            batch_asr_results = self.asr.transcribe(batch_asr_uttid, batch_asr_wav)
            logger.info(f"ASR: {batch_asr_results}")
            batch_asr_results = [a for a in batch_asr_results if not re.search(r"(<blank>)|(<sil>)", a["text"])]
            asr_results.extend(batch_asr_results)

            if self.config.enable_lid:
                batch_lid_results = self.lid.process(batch_asr_uttid, batch_asr_wav)
                logger.info(f"LID: {batch_lid_results}")
            else:
                batch_lid_results = [None] * len(batch_asr_results)
            lid_results.extend(batch_lid_results)

            batch_asr_uttid = []
            batch_asr_wav = []

        # 4. ASR output to Postprocess input
        if self.config.enable_punc:
            punc_results = []
            batch_asr_text = []
            batch_asr_uttid = []
            batch_asr_timestamp = []
            for j, asr_result in enumerate(asr_results):
                batch_asr_text.append(asr_result["text"])
                batch_asr_uttid.append(asr_result["uttid"])
                if "timestamp" in asr_result:
                    batch_asr_timestamp.append(asr_result["timestamp"])
                if len(batch_asr_text) < self.config.punc_batch_size and j != len(asr_results) - 1:
                    continue

                # 5. Punc
                if self.config.asr_config.return_timestamp:
                    batch_punc_results = self.punc.process_with_timestamp(batch_asr_timestamp, batch_asr_uttid)
                else:
                    batch_punc_results = self.punc.process(batch_asr_text, batch_asr_uttid)
                logger.info(f"Punc: {batch_punc_results}")

                punc_results.extend(batch_punc_results)
                batch_asr_text = []
                batch_asr_uttid = []
                batch_asr_timestamp = []
        else:
            punc_results = asr_results

        # 6. Put all together & Format
        sentences = []
        words = []
        for asr_result, punc_result, lid_result in zip(asr_results, punc_results, lid_results):
            assert asr_result["uttid"] == punc_result["uttid"], f"fix code: {asr_result} | {punc_result}"
            start_ms, end_ms = asr_result["uttid"].split("_")[-2:]
            assert start_ms.startswith("s") and end_ms.startswith("e")
            start_ms, end_ms = int(start_ms[1:]), int(end_ms[1:])
            if self.config.asr_config.return_timestamp:
                sub_sentences = []
                if self.config.enable_punc:
                    for i, punc_sent in enumerate(punc_result["punc_sentences"]):
                        start = start_ms + int(punc_sent["start_s"]*1000)
                        end = start_ms + int(punc_sent["end_s"]*1000)
                        if i == 0:
                            start = start_ms
                        if i == len(punc_result["punc_sentences"]) - 1:
                            end = end_ms
                        sub_sentence = {
                            "start_ms": start,
                            "end_ms": end,
                            "text": punc_sent["punc_text"],
                            "asr_confidence": asr_result["confidence"],
                            "lang": None,
                            "lang_confidence": 0
                        }
                        if lid_result:
                            sub_sentence["lang"] = lid_result["lang"]
                            sub_sentence["lang_confidence"] = lid_result["confidence"]
                        sub_sentences.append(sub_sentence)
                else:
                    sub_sentences = [{
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": asr_result["text"],
                        "asr_confidence": asr_result["confidence"],
                        "lang": None,
                        "lang_confidence": 0
                    }]
                sentences.extend(sub_sentences)
            else:
                text = punc_result["punc_text"] if self.config.enable_punc else asr_result["text"]
                sentence = {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "text": text,
                    "asr_confidence": asr_result["confidence"],
                    "lang": None,
                    "lang_confidence": 0
                }
                if lid_result:
                    sentence["lang"] = lid_result["lang"]
                    sentence["lang_confidence"] = lid_result["confidence"]
                sentences.append(sentence)
            
            if "timestamp" in asr_result:
                for w, s, e in asr_result["timestamp"]:
                    word = {"start_ms": int(s*1000+start_ms), "end_ms":int(e*1000+start_ms), "text": w}
                    words.append(word)

        vad_segments_ms = [(int(s*1000), int(e*1000)) for s, e in vad_result["timestamps"]]
        text = "".join(s["text"] for s in sentences)

        result = {
            "uttid": uttid,
            "text": text,
            "sentences": sentences,
            "vad_segments_ms": vad_segments_ms,
            "dur_s": dur,
            "words": words,
            "wav_path": wav_path
        }
        return result
