import gc

import numpy as np
from faster_whisper import WhisperModel

from bragi.adapters.stt import STTAdapter, Segment, TranscriptResult, Word

WHISPER_MODEL_SIZES = {
    "tiny", "tiny.en", "base", "base.en", "small", "small.en",
    "medium", "medium.en", "large", "large-v1", "large-v2", "large-v3",
    "large-v3-turbo", "turbo", "distil-large-v2", "distil-large-v3",
    "distil-medium.en", "distil-small.en",
}

WHISPER_LANGUAGES = [
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
    "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
    "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "yue", "zh",
]


class FasterWhisperAdapter(STTAdapter):

    def __init__(self) -> None:
        self._model: WhisperModel | None = None

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "whisper" in repo or repo in WHISPER_MODEL_SIZES

    def load(self, model_path: str, device: str, **kwargs) -> None:
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        compute_type = kwargs.get("compute_type") or "default"
        self._model = WhisperModel(model_path, device=device, compute_type=compute_type)

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
        return self._run(audio, language=language, temperature=temperature, word_timestamps=word_timestamps, task="transcribe")

    def translate(self, audio: np.ndarray, temperature: float) -> TranscriptResult:
        return self._run(audio, language=None, temperature=temperature, word_timestamps=False, task="translate")

    def _run(
        self,
        audio: np.ndarray,
        *,
        language: str | None,
        temperature: float,
        word_timestamps: bool,
        task: str,
    ) -> TranscriptResult:
        segments_gen, info = self._model.transcribe(
            audio,
            language=language,
            temperature=temperature,
            word_timestamps=word_timestamps,
            task=task,
        )

        raw_segments = list(segments_gen)

        segments = [
            Segment(
                id=s.id,
                start=s.start,
                end=s.end,
                text=s.text,
                tokens=list(s.tokens) if s.tokens else None,
                temperature=s.temperature,
                avg_logprob=s.avg_logprob,
                compression_ratio=s.compression_ratio,
                no_speech_prob=s.no_speech_prob,
            )
            for s in raw_segments
        ]

        words = None
        if word_timestamps:
            words = []
            for s in raw_segments:
                if s.words:
                    for w in s.words:
                        words.append(Word(word=w.word, start=w.start, end=w.end))

        text = " ".join(s.text.strip() for s in raw_segments)

        return TranscriptResult(
            text=text,
            language=info.language,
            duration=info.duration,
            segments=segments,
            words=words,
        )

    def get_supported_languages(self) -> list[str]:
        return WHISPER_LANGUAGES

    def get_sample_rate(self) -> int:
        return 16000

    def supports_translation(self) -> bool:
        return True

    def supports_streaming(self) -> bool:
        return False
