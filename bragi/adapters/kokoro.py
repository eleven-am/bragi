import gc
from typing import AsyncIterator

import numpy as np
from kokoro import KPipeline

from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio

KOKORO_VOICES = [
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
    "ef_dora",
    "em_alex", "em_santa",
    "ff_siwis",
    "hf_alpha", "hf_beta",
    "hm_omega", "hm_psi",
    "if_sara",
    "im_nicola",
    "jf_alpha", "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_beta", "jm_kumo",
    "pf_dora",
    "pm_alex", "pm_santa",
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao",
    "zm_yunjian", "zm_yunxi", "zm_yunyang",
]


class KokoroAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._pipeline: KPipeline | None = None

    @staticmethod
    def detect(config: dict) -> bool:
        return "kokoro" in config.get("repo", "").lower()

    def load(self, model_path: str, device: str, **kwargs) -> None:
        self._pipeline = KPipeline(lang_code="a")

    def unload(self) -> None:
        del self._pipeline
        self._pipeline = None
        gc.collect()

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        chunks = []
        for _graphemes, _phonemes, audio_np in self._pipeline(text, voice=voice, speed=speed):
            if audio_np is not None:
                chunks.append(audio_np)

        if not chunks:
            return np.array([], dtype=np.float32), 24000

        return np.concatenate(chunks), 24000

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        audio, sr = self.synthesize_raw(text, voice, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        for _graphemes, _phonemes, audio_np in self._pipeline(text, voice=voice, speed=speed):
            if audio_np is not None:
                chunk, _content_type = encode_audio(audio_np, 24000, response_format)
                yield chunk

    def get_available_voices(self) -> list[str]:
        return KOKORO_VOICES

    def get_sample_rate(self) -> int:
        return 24000

    def supports_streaming(self) -> bool:
        return True

    def supports_voice_cloning(self) -> bool:
        return False

    def synthesize_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float, response_format: str
    ) -> bytes:
        raise NotImplementedError("Kokoro does not support voice cloning with reference audio")
