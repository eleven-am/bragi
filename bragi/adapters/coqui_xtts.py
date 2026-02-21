import gc
import tempfile
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio


class CoquiXTTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._tts = None
        self._speakers: list[str] = []

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "xtts" in repo or "coqui" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from TTS.api import TTS

        self._tts = TTS(model_name=model_path)
        self._tts.to(device)
        self._speakers = self._tts.speakers or []

    def unload(self) -> None:
        del self._tts
        self._tts = None
        self._speakers = []
        gc.collect()

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        wav = self._tts.tts(text=text, speaker=voice, language="en")
        return np.array(wav, dtype=np.float32), 24000

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        audio, sr = self.synthesize_raw(text, voice, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Coqui XTTS streaming is not supported via this adapter")
        yield  # noqa: unreachable — required to make this an async generator

    def get_available_voices(self) -> list[str]:
        return self._speakers or ["default"]

    def get_sample_rate(self) -> int:
        return 24000

    def supports_streaming(self) -> bool:
        return False

    def supports_voice_cloning(self) -> bool:
        return True

    def synthesize_raw_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float
    ) -> tuple[np.ndarray, int]:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(reference_audio)

            wav = self._tts.tts(text=text, speaker_wav=tmp_path, language="en")
            return np.array(wav, dtype=np.float32), 24000
        finally:
            if tmp_path:
                import os
                os.unlink(tmp_path)

    def synthesize_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float, response_format: str
    ) -> bytes:
        audio, sr = self.synthesize_raw_with_reference(text, reference_audio, transcript, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded
