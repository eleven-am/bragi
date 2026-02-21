import io

import numpy as np
import soundfile as sf

CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
}


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    import lameenc

    int16_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(sample_rate)
    encoder.set_channels(1)
    encoder.set_quality(2)
    return bytes(encoder.encode(int16_audio.tobytes()) + encoder.flush())


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray, sample_rate: int) -> bytes:
    int16_audio = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    return int16_audio.tobytes()


def _encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC")
    return buf.getvalue()


def _encode_opus(audio: np.ndarray, sample_rate: int) -> bytes:
    raise ValueError(
        "Opus encoding requires opuslib. Install with: pip install opuslib"
    )


_ENCODERS = {
    "mp3": _encode_mp3,
    "wav": _encode_wav,
    "pcm": _encode_pcm,
    "flac": _encode_flac,
    "opus": _encode_opus,
}


def encode_audio(
    audio: np.ndarray, sample_rate: int, format: str
) -> tuple[bytes, str]:
    encoder = _ENCODERS.get(format)
    if encoder is None:
        raise ValueError(
            f"Unsupported output format: {format}. "
            f"Supported formats: {', '.join(sorted(_ENCODERS.keys()))}"
        )
    content_type = CONTENT_TYPES[format]
    return encoder(audio, sample_rate), content_type
