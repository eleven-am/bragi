import gc
import tempfile

import numpy as np
import soundfile as sf
from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult, Word


class ParakeetAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "parakeet" in repo or "nemotron-speech" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        import nemo.collections.asr as nemo_asr

        self._model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_path)
        if device != "cpu":
            self._model = self._model.to(device)

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
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                sf.write(tmp_path, audio, 16000, format="WAV", subtype="PCM_16")

            hypotheses = self._model.transcribe([tmp_path])

            if isinstance(hypotheses, list) and len(hypotheses) > 0:
                hyp = hypotheses[0]
                text = hyp.text if hasattr(hyp, "text") else str(hyp)
            else:
                text = str(hypotheses)

            words = None
            if word_timestamps and hasattr(hyp, "timestep") and hyp.timestep:
                words = [
                    Word(word=w.word, start=w.start, end=w.end)
                    for w in hyp.timestep
                    if hasattr(w, "word")
                ]

            return TranscriptResult(
                text=text,
                language=language,
                duration=len(audio) / 16000,
                segments=[Segment(id=0, start=0.0, end=len(audio) / 16000, text=text)] if text else None,
                words=words if words else None,
            )
        finally:
            if tmp_path:
                import os
                os.unlink(tmp_path)

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        raise NotImplementedError("Parakeet does not support translation")

    def get_supported_languages(self) -> list[str]:
        return [
            "en", "de", "fr", "es", "it", "pt", "nl", "ja", "ko", "zh",
            "hi", "ar", "ru", "uk", "pl", "sv", "fi", "no", "da", "ca",
            "cs", "hr", "hu", "ro", "sk",
        ]

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False
