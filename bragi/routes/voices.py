from fastapi import APIRouter, Form, Request, UploadFile

from bragi.schemas.errors import (
    InvalidModelError,
    InvalidVoiceError,
    VoiceCloningNotSupportedError,
    VoiceConflictError,
)
from bragi.schemas.voices import VoiceCreateResponse, VoiceListResponse, VoiceObject

router = APIRouter()


@router.get("/audio/voices")
async def list_voices(request: Request) -> VoiceListResponse:
    registry = request.app.state.registry
    voice_store = request.app.state.voice_store

    voices: list[VoiceObject] = []

    for voice_name, model_alias in registry.list_all_voices():
        voices.append(
            VoiceObject(
                id=voice_name,
                name=voice_name,
                model=model_alias,
                custom=False,
                languages=[],
            )
        )

    for cv in await voice_store.list_all():
        voices.append(
            VoiceObject(
                id=cv.id,
                name=cv.name,
                model=cv.adapter_alias,
                custom=True,
                languages=[],
            )
        )

    return VoiceListResponse(data=voices)


@router.post("/audio/voices")
async def create_voice(
    request: Request,
    file: UploadFile,
    name: str = Form(...),
    transcript: str = Form(...),
    model: str | None = Form(None),
) -> VoiceCreateResponse:
    registry = request.app.state.registry
    voice_store = request.app.state.voice_store

    if registry.has_voice(name) or await voice_store.get_by_name(name) is not None:
        raise VoiceConflictError(name)

    adapter_alias = model or ""

    if model:
        if not registry.has_model(model):
            raise InvalidModelError(model)
        adapter = registry.get_tts(model)
        if not adapter.supports_voice_cloning():
            raise VoiceCloningNotSupportedError(model)

    audio_data = await file.read()
    original_filename = file.filename or "reference.wav"

    cv = await voice_store.create(
        name=name,
        transcript=transcript,
        audio_data=audio_data,
        original_filename=original_filename,
        adapter_alias=adapter_alias,
    )

    if adapter_alias:
        registry.register_custom_voice(cv.name, adapter_alias)

    return VoiceCreateResponse(
        id=cv.id,
        name=cv.name,
        model=cv.adapter_alias,
        created_at=cv.created_at,
    )


@router.delete("/audio/voices/{voice_id}")
async def delete_voice(request: Request, voice_id: str):
    voice_store = request.app.state.voice_store
    registry = request.app.state.registry

    cv = await voice_store.get_by_id(voice_id)
    if cv is None:
        raise InvalidVoiceError(voice_id)

    registry.unregister_voice(cv.name)
    await voice_store.delete(voice_id)

    return {"deleted": True, "id": voice_id}
