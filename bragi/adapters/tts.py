from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np


class TTSAdapter(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes: ...

    @abstractmethod
    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]: ...

    @abstractmethod
    def get_available_voices(self) -> list[str]: ...

    @abstractmethod
    def get_sample_rate(self) -> int: ...

    @abstractmethod
    def supports_streaming(self) -> bool: ...

    @abstractmethod
    def supports_voice_cloning(self) -> bool: ...

    @abstractmethod
    def synthesize_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float, response_format: str
    ) -> bytes: ...

    @staticmethod
    @abstractmethod
    def detect(config: dict) -> bool: ...

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        pcm_bytes = self.synthesize(text, voice, speed, "pcm")
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, self.get_sample_rate()

    def synthesize_raw_with_reference(
        self, text: str, reference_audio: bytes, transcript: str, speed: float
    ) -> tuple[np.ndarray, int]:
        pcm_bytes = self.synthesize_with_reference(text, reference_audio, transcript, speed, "pcm")
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
        return audio, self.get_sample_rate()
