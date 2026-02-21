import gc

import numpy as np
from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult


class SpeechBrainAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        return "speechbrain" in config.get("repo", "").lower()

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from speechbrain.inference.ASR import EncoderASR

        self._model = EncoderASR.from_hparams(source=model_path, run_opts={"device": device})

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
        import torch

        wavs = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        wav_lens = torch.tensor([1.0])

        predicted_words, _ = self._model.transcribe_batch(wavs, wav_lens)
        text = predicted_words[0] if predicted_words else ""

        return TranscriptResult(
            text=text,
            language=language,
            duration=len(audio) / 16000,
            segments=[Segment(id=0, start=0.0, end=len(audio) / 16000, text=text)] if text else None,
            words=None,
        )

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        raise NotImplementedError("SpeechBrain does not support translation")

    def get_supported_languages(self) -> list[str]:
        return ["en"]

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return False

    def supports_streaming(self) -> bool:
        return False
