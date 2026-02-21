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
)
from bragi.schemas.responses import (
    SegmentResponse,
    TranscriptionResponse,
    TranscriptionVerboseResponse,
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


def _result_to_words(result: TranscriptResult) -> list[WordResponse]:
    if not result.words:
        return []
    return [
        WordResponse(word=w.word, start=w.start, end=w.end)
        for w in result.words
    ]


@router.post("/audio/transcriptions")
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    stream: bool = Form(False),
    timestamp_granularities: list[str] | None = Form(None, alias="timestamp_granularities[]"),
):
    registry = request.app.state.registry
    config = request.app.state.config

    if not registry.has_model(model):
        raise InvalidModelError(model)

    try:
        adapter = registry.get_stt(model)
    except KeyError:
        raise ModelNotLoadedError(model)

    data = await file.read()

    max_size = parse_file_size(config.server.max_file_size)
    if len(data) > max_size:
        raise FileTooLargeError(config.server.max_file_size)

    try:
        audio = decode_audio(data, file.filename)
    except ValueError:
        raise InvalidFileFormatError()

    word_timestamps = bool(
        timestamp_granularities and "word" in timestamp_granularities
    )

    result = adapter.transcribe(
        audio=audio,
        language=language,
        temperature=temperature,
        word_timestamps=word_timestamps,
    )

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
            task="transcribe",
            language=result.language or "",
            duration=result.duration,
            text=result.text,
            segments=_result_to_segments(result),
            words=_result_to_words(result),
        )

    return TranscriptionResponse(text=result.text)
