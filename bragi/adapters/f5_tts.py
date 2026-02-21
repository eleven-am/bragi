import gc
import tempfile
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio


class F5TTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._tts = None

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "f5-tts" in repo or "f5tts" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from f5_tts.api import F5TTS

        self._tts = F5TTS(model_type="F5-TTS")

    def unload(self) -> None:
        del self._tts
        self._tts = None
        gc.collect()

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        raise NotImplementedError("F5-TTS requires reference audio. Use a custom voice.")

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("F5-TTS does not support streaming")
        yield  # noqa: unreachable — required to make this an async generator

    def get_available_voices(self) -> list[str]:
        return []

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

            wav, sr, _ = self._tts.infer(ref_file=tmp_path, ref_text=transcript, gen_text=text)

            audio = np.array(wav, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.squeeze()

            if sr != 24000:
                import soxr
                audio = soxr.resample(audio, sr, 24000)

            return audio, 24000
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
