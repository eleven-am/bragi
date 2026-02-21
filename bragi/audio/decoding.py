import io
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import soxr

TARGET_SAMPLE_RATE = 16000

SOUNDFILE_FORMATS = {"wav", "flac", "ogg"}
FFMPEG_FORMATS = {"mp3", "mp4", "m4a", "webm", "mpeg", "mpga"}
SUPPORTED_FORMATS = SOUNDFILE_FORMATS | FFMPEG_FORMATS

EXTENSION_MAP = {
    "flac": "flac",
    "mp3": "mp3",
    "mp4": "mp4",
    "mpeg": "mp3",
    "mpga": "mp3",
    "m4a": "m4a",
    "ogg": "ogg",
    "wav": "wav",
    "webm": "webm",
}


def _get_format(filename: str | None) -> str | None:
    if not filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else None
    return EXTENSION_MAP.get(ext)


def _decode_soundfile(data: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    return audio, sr


def _decode_ffmpeg(data: bytes, fmt: str) -> tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=f".{fmt}", delete=True) as tmp:
        tmp.write(data)
        tmp.flush()

        result = subprocess.run(
            [
                "ffmpeg", "-i", tmp.name,
                "-f", "wav",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(TARGET_SAMPLE_RATE),
                "-y", "pipe:1",
            ],
            capture_output=True,
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode(errors="replace"))

        audio, sr = sf.read(io.BytesIO(result.stdout), dtype="float32")
        return audio, sr


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _resample(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == TARGET_SAMPLE_RATE:
        return audio
    return soxr.resample(audio, sr, TARGET_SAMPLE_RATE)


def decode_audio(data: bytes, filename: str | None = None) -> np.ndarray:
    fmt = _get_format(filename)

    if fmt and fmt not in SUPPORTED_FORMATS and fmt not in EXTENSION_MAP.values():
        raise ValueError(
            f"Unsupported audio format: {fmt}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}"
        )

    audio = None
    sr = None

    if fmt is None or fmt in SOUNDFILE_FORMATS:
        try:
            audio, sr = _decode_soundfile(data)
        except Exception:
            if fmt in SOUNDFILE_FORMATS:
                raise

    if audio is None:
        ffmpeg_fmt = fmt or "mp3"
        try:
            audio, sr = _decode_ffmpeg(data, ffmpeg_fmt)
        except Exception as e:
            raise ValueError(f"Failed to decode audio: {e}") from e

    audio = _to_mono(audio)
    audio = _resample(audio, sr)
    return audio.astype(np.float32)
