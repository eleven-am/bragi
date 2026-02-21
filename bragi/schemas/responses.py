from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str


class SegmentResponse(BaseModel):
    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: list[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class WordResponse(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionVerboseResponse(BaseModel):
    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[SegmentResponse] = []
    words: list[WordResponse] = []


class TranslationResponse(BaseModel):
    text: str


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "bragi"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject] = []


def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def format_srt(segments: list[SegmentResponse]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


def format_vtt(segments: list[SegmentResponse]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)
