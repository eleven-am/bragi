import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response

from bragi.audio.chunking import chunk_text
from bragi.audio.encoding import CONTENT_TYPES, encode_audio
from bragi.schemas.errors import InvalidModelError, InvalidVoiceError, ModelNotLoadedError
from bragi.schemas.requests import SpeechRequest

router = APIRouter()


@router.post("/audio/speech")
async def create_speech(request: Request, body: SpeechRequest):
    registry = request.app.state.registry
    voice_store = request.app.state.voice_store

    custom_voice = await voice_store.get_by_name(body.voice)

    if body.model:
        if not registry.has_model(body.model):
            raise InvalidModelError(body.model)
        try:
            adapter = registry.get_tts(body.model)
        except KeyError:
            raise ModelNotLoadedError(body.model)
    elif custom_voice and custom_voice.adapter_alias:
        try:
            adapter = registry.get_tts(custom_voice.adapter_alias)
        except KeyError:
            raise ModelNotLoadedError(custom_voice.adapter_alias)
    else:
        try:
            _, adapter = registry.get_tts_by_voice(body.voice)
        except KeyError:
            raise InvalidVoiceError(body.voice)

    chunks = chunk_text(body.input)

    if custom_voice:
        reference_audio = voice_store.get_reference_audio(custom_voice.id)
        audio_arrays = []
        sample_rate = None
        for chunk in chunks:
            audio, sr = adapter.synthesize_raw_with_reference(
                text=chunk,
                reference_audio=reference_audio,
                transcript=custom_voice.transcript,
                speed=body.speed,
            )
            audio_arrays.append(audio)
            sample_rate = sr
    else:
        available_voices = adapter.get_available_voices()
        if available_voices and body.voice not in available_voices:
            raise InvalidVoiceError(body.voice)

        audio_arrays = []
        sample_rate = None
        for chunk in chunks:
            audio, sr = adapter.synthesize_raw(
                text=chunk,
                voice=body.voice,
                speed=body.speed,
            )
            audio_arrays.append(audio)
            sample_rate = sr

    combined_audio = np.concatenate(audio_arrays)
    audio_bytes, content_type = encode_audio(combined_audio, sample_rate, body.response_format)

    return Response(content=audio_bytes, media_type=content_type)
