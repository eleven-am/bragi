from __future__ import annotations

from dataclasses import dataclass

from bragi.adapters.stt import STTAdapter
from bragi.adapters.tts import TTSAdapter


@dataclass
class ModelInfo:
    alias: str
    model_type: str
    repo: str | None
    device: str | None
    status: str


class ModelRegistry:
    def __init__(self) -> None:
        self._stt_adapters: dict[str, STTAdapter] = {}
        self._tts_adapters: dict[str, TTSAdapter] = {}
        self._model_info: dict[str, ModelInfo] = {}
        self._voice_to_tts: dict[str, tuple[str, TTSAdapter]] = {}

    def register_stt(self, alias: str, adapter: STTAdapter, info: ModelInfo) -> None:
        self._stt_adapters[alias] = adapter
        self._model_info[alias] = info

    def register_tts(self, alias: str, adapter: TTSAdapter, info: ModelInfo) -> None:
        self._tts_adapters[alias] = adapter
        self._model_info[alias] = info

        for voice in adapter.get_available_voices():
            if voice not in self._voice_to_tts:
                self._voice_to_tts[voice] = (alias, adapter)

    def get_stt(self, alias: str) -> STTAdapter:
        if alias not in self._stt_adapters:
            raise KeyError(f"STT model not found: {alias!r}")
        return self._stt_adapters[alias]

    def get_tts(self, alias: str) -> TTSAdapter:
        if alias not in self._tts_adapters:
            raise KeyError(f"TTS model not found: {alias!r}")
        return self._tts_adapters[alias]

    def get_tts_by_voice(self, voice: str) -> tuple[str, TTSAdapter]:
        if voice not in self._voice_to_tts:
            raise KeyError(f"No adapter found for voice: {voice!r}")
        return self._voice_to_tts[voice]

    def list_all_voices(self) -> list[tuple[str, str]]:
        return [(voice, alias) for voice, (alias, _) in self._voice_to_tts.items()]

    def register_custom_voice(self, voice_name: str, alias: str) -> None:
        if alias not in self._tts_adapters:
            return
        self._voice_to_tts[voice_name] = (alias, self._tts_adapters[alias])

    def unregister_voice(self, voice_name: str) -> None:
        self._voice_to_tts.pop(voice_name, None)

    def has_voice(self, voice_name: str) -> bool:
        return voice_name in self._voice_to_tts

    def has_model(self, alias: str) -> bool:
        return alias in self._model_info

    def list_models(self) -> list[ModelInfo]:
        return list(self._model_info.values())

    def unload_all(self) -> None:
        for adapter in self._stt_adapters.values():
            adapter.unload()
        for adapter in self._tts_adapters.values():
            adapter.unload()
        self._stt_adapters.clear()
        self._tts_adapters.clear()
        self._model_info.clear()
        self._voice_to_tts.clear()
