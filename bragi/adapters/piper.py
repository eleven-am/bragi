import gc
import io
import wave
from typing import AsyncIterator

import numpy as np
import soundfile as sf
from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio


class PiperAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._voice = None
        self._sample_rate: int = 22050

    @staticmethod
    def detect(config: dict) -> bool:
        return "piper" in config.get("repo", "").lower()

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from piper import PiperVoice

        self._voice = PiperVoice.load(model_path)
        self._sample_rate = self._voice.config.sample_rate

    def unload(self) -> None:
        del self._voice
        self._voice = None
        gc.collect()

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self._sample_rate)
            self._voice.synthesize(text, wav_file, length_scale=1.0 / speed)

        buf.seek(0)
        audio, sr = sf.read(buf, dtype="float32")
        return audio, sr

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        audio, sr = self.synthesize_raw(text, voice, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        for chunk in self._voice.synthesize_stream_raw(text, length_scale=1.0 / speed):
            audio = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767.0
            encoded, _ = encode_audio(audio, self._sample_rate, response_format)
            yield encoded

    def get_available_voices(self) -> list[str]:
        return ["default"]

    def get_sample_rate(self) -> int:
        return self._sample_rate

    def supports_streaming(self) -> bool:
        return True

    def supports_voice_cloning(self) -> bool:
        return False

    def synthesize_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float, response_format: str
    ) -> bytes:
        raise NotImplementedError("Piper does not support voice cloning with reference audio")
