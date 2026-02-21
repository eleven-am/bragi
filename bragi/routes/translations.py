from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import PlainTextResponse

from bragi.adapters.stt import TranscriptResult
from bragi.audio.decoding import decode_audio
from bragi.config import parse_file_size
from bragi.schemas.errors import (
    FileTooLargeError,
    InvalidFileFormatError,
    InvalidModelError,
    ModelNotLoadedError,
    UnsupportedFeatureError,
)
from bragi.schemas.responses import (
    SegmentResponse,
    TranscriptionVerboseResponse,
    TranslationResponse,
    WordResponse,
    format_srt,
    format_vtt,
)

router = APIRouter()


def _result_to_segments(result: TranscriptResult) -> list[SegmentResponse]:
    if not result.segments:
        return []
    return [
        SegmentResponse(
            id=seg.id,
            start=seg.start,
            end=seg.end,
            text=seg.text,
            tokens=seg.tokens or [],
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
        )
        for seg in result.segments
    ]


@router.post("/audio/translations")
async def create_translation(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    registry = request.app.state.registry
    config = request.app.state.config

    if not registry.has_model(model):
        raise InvalidModelError(model)

    try:
        adapter = registry.get_stt(model)
    except KeyError:
        raise ModelNotLoadedError(model)

    if not adapter.supports_translation():
        raise UnsupportedFeatureError("translation", model)

    data = await file.read()

    max_size = parse_file_size(config.server.max_file_size)
    if len(data) > max_size:
        raise FileTooLargeError(config.server.max_file_size)

    try:
        audio = decode_audio(data, file.filename)
    except ValueError:
        raise InvalidFileFormatError()

    result = adapter.translate(audio=audio, temperature=temperature)

    if response_format == "text":
        return PlainTextResponse(result.text)

    if response_format == "srt":
        segments = _result_to_segments(result)
        return PlainTextResponse(format_srt(segments), media_type="text/plain")

    if response_format == "vtt":
        segments = _result_to_segments(result)
        return PlainTextResponse(format_vtt(segments), media_type="text/vtt")

    if response_format == "verbose_json":
        return TranscriptionVerboseResponse(
            task="translate",
            language=result.language or "",
            duration=result.duration,
            text=result.text,
            segments=_result_to_segments(result),
        )

    return TranslationResponse(text=result.text)
