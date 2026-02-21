import gc

import numpy as np
from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult


class MoonshineAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        return "moonshine" in config.get("repo", "").lower()

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from moonshine_onnx import MoonshineOnnxModel

        self._model = MoonshineOnnxModel(model_name=model_path)

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
        output = self._model.generate(audio)
        text = output[0] if isinstance(output, list) else output

        return TranscriptResult(
            text=text,
            language="en",
            duration=len(audio) / 16000,
            segments=[Segment(id=0, start=0.0, end=len(audio) / 16000, text=text)] if text else None,
            words=None,
        )

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        raise NotImplementedError("Moonshine does not support translation")

    def get_supported_languages(self) -> list[str]:
        return ["en"]

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False
