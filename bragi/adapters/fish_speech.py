import gc
import tempfile
from typing import AsyncIterator

import numpy as np
from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio


class FishSpeechAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model = None
        self._device: str = "cpu"

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "fish-speech" in repo or "fishaudio" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from fish_speech.inference import load_model

        self._device = device
        self._model = load_model(model_path, device=device)

    def unload(self) -> None:
        del self._model
        self._model = None
        gc.collect()

    def _infer(self, text: str, reference_path: str | None = None) -> np.ndarray:
        if reference_path:
            audio = self._model(text, reference=reference_path)
        else:
            audio = self._model(text)

        if not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        if audio.ndim > 1:
            audio = audio.squeeze()

        return audio.astype(np.float32)

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        return self._infer(text), 44100

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        audio, sr = self.synthesize_raw(text, voice, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Fish Speech does not support streaming via this adapter")
        yield  # noqa: unreachable — required to make this an async generator

    def get_available_voices(self) -> list[str]:
        return ["default"]

    def get_sample_rate(self) -> int:
        return 44100

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

            return self._infer(text, reference_path=tmp_path), 44100
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
