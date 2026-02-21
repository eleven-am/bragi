import gc

import numpy as np
from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult, Word


class ParaformerAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "paraformer" in repo or "funasr" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from funasr import AutoModel

        self._model = AutoModel(model=model_path, device=device)

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
        result = self._model.generate(input=audio, batch_size_s=0)

        full_text_parts = []
        segments = []
        words = []

        for idx, item in enumerate(result):
            text = item.get("text", "")
            full_text_parts.append(text)
            segments.append(Segment(id=idx, start=0.0, end=0.0, text=text))

            if word_timestamps and "timestamp" in item:
                for entry in item["timestamp"]:
                    if len(entry) == 3:
                        words.append(Word(word=entry[0], start=entry[1] / 1000.0, end=entry[2] / 1000.0))

        return TranscriptResult(
            text=" ".join(full_text_parts),
            language=language,
            duration=len(audio) / 16000,
            segments=segments if segments else None,
            words=words if words else None,
        )

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        raise NotImplementedError("Paraformer does not support translation")

    def get_supported_languages(self) -> list[str]:
        return ["zh", "en"]

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False
