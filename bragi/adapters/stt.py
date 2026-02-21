from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator
import numpy as np


@dataclass
class Word:
    word: str
    start: float
    end: float


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    tokens: list[int] | None = None
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


@dataclass
class TranscriptResult:
    text: str
    language: str | None = None
    duration: float = 0.0
    segments: list[Segment] | None = None
    words: list[Word] | None = None


class STTAdapter(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str, **kwargs) -> None: ...

    @abstractmethod
    def unload(self) -> None: ...

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> TranscriptResult: ...

    @abstractmethod
    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult: ...

    @abstractmethod
    def get_supported_languages(self) -> list[str]: ...

    @abstractmethod
    def get_sample_rate(self) -> int: ...

    @abstractmethod
    def supports_translation(self) -> bool: ...

    @abstractmethod
    def supports_streaming(self) -> bool: ...

    @staticmethod
    @abstractmethod
    def detect(config: dict) -> bool: ...
