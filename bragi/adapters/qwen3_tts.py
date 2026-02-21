import gc
import tempfile
from typing import AsyncIterator

import numpy as np
from bragi.adapters.tts import TTSAdapter
from bragi.audio.encoding import encode_audio

QWEN3_VOICES = [
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
]


def _speed_to_instruct(speed: float) -> str:
    if speed <= 0.5:
        return "Speak very slowly"
    if speed <= 0.8:
        return "Speak slowly"
    if speed <= 1.2:
        return "Speak at a normal pace"
    if speed < 1.5:
        return "Speak quickly"
    return "Speak very quickly"


class Qwen3TTSAdapter(TTSAdapter):

    def __init__(self) -> None:
        self._model = None

    @staticmethod
    def detect(config: dict) -> bool:
        repo = config.get("repo", "").lower()
        return "qwen" in repo and "tts" in repo

    def load(self, model_path: str, device: str, **kwargs) -> None:
        from qwen_tts import Qwen3TTSModel

        self._model = Qwen3TTSModel.from_pretrained(model_path)

    def unload(self) -> None:
        del self._model
        self._model = None
        gc.collect()

    def synthesize_raw(self, text: str, voice: str, speed: float) -> tuple[np.ndarray, int]:
        instruct = _speed_to_instruct(speed)
        speaker = voice if voice in QWEN3_VOICES else QWEN3_VOICES[0]

        wavs, sr = self._model.generate_custom_voice(
            text=text,
            language="Auto",
            speaker=speaker,
            instruct=instruct,
        )

        return wavs.squeeze().cpu().numpy().astype(np.float32), 24000

    def synthesize(self, text: str, voice: str, speed: float, response_format: str) -> bytes:
        audio, sr = self.synthesize_raw(text, voice, speed)
        encoded, _ = encode_audio(audio, sr, response_format)
        return encoded

    async def synthesize_stream(
        self, text: str, voice: str, speed: float, response_format: str
    ) -> AsyncIterator[bytes]:
        raise NotImplementedError("Qwen3 TTS streaming is not yet supported via this adapter")
        yield  # noqa: unreachable — required to make this an async generator

    def get_available_voices(self) -> list[str]:
        return list(QWEN3_VOICES)

    def get_sample_rate(self) -> int:
        return 24000

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

            wavs, sr = self._model.generate_voice_clone(
                text=text,
                language="Auto",
                ref_audio=tmp_path,
                ref_text=transcript,
            )

            return wavs.squeeze().cpu().numpy().astype(np.float32), 24000
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
