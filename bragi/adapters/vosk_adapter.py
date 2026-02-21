import gc
import json

import numpy as np
from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult, Word


class VoskAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        return "vosk" in config.get("repo", "").lower()

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        self._model = Model(model_path=model_path)

    def unload(self) -> None:
        del self._model
        self._model = None
        gc.collect()

    def transcribe(
        self,
        audio: np.ndarray,
        language: str | None,
        temperature: float,
        word_timestamps: bool,
    ) -> TranscriptResult:
        from vosk import KaldiRecognizer

        rec = KaldiRecognizer(self._model, 16000)
        if word_timestamps:
            rec.SetWords(True)

        int16_data = (audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes()
        rec.AcceptWaveform(int16_data)
        result = json.loads(rec.FinalResult())

        text = result.get("text", "")
        words = None
        segments = None

        if word_timestamps and "result" in result:
            words = [
                Word(word=w["word"], start=w["start"], end=w["end"])
                for w in result["result"]
            ]

        if text:
            duration = len(audio) / 16000
            segments = [Segment(id=0, start=0.0, end=duration, text=text)]

        return TranscriptResult(
            text=text,
            language=language,
            duration=len(audio) / 16000,
            segments=segments,
            words=words,
        )

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        raise NotImplementedError("Vosk does not support translation")

    def get_supported_languages(self) -> list[str]:
        return [
            "en", "de", "fr", "es", "pt", "it", "nl", "ca", "uk", "kk",
            "ja", "zh", "ar", "hi", "ko", "fa", "vi", "tl", "uz", "tr",
        ]

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return True
