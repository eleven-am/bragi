# Bragi — OpenAI-Compatible Audio API Spec

> Self-hosted, open-source STT/TTS server with a growing adapter ecosystem.
> Drop-in replacement for OpenAI's `/v1/audio/*` endpoints.
> Swap models within supported families by changing one line in your config.

---

## Stack

- **Language**: Python
- **Framework**: FastAPI
- **Containerization**: Docker
- **Model Source**: Hugging Face Hub (auto-download)
- **STT/TTS Engines**: Adapter-based architecture with a clean plugin interface

---

## Core Philosophy

Bragi makes self-hosted STT/TTS simple — but it's honest about what that means.

**What Bragi does:**

- Provides an OpenAI-compatible API so your existing code works with zero changes
- Ships with adapters for the most popular model families out of the box
- Lets you swap models freely within a supported family (e.g. `whisper-large-v3` → `whisper-tiny` → `distil-whisper`) by changing one config line
- Auto-downloads models from Hugging Face with a single global token
- Handles device management, OOM fallback, and model lifecycle

**What "plug and play" actually means:**

- Within a supported adapter family: truly plug and play. Change the repo name, restart, done.
- Across different architectures: requires a new adapter. Each adapter is ~100-200 lines of Python that bridges a model's inference API to Bragi's standard interface.
- Arbitrary HF models: not automatic. If nobody has written an adapter for that architecture, Bragi will tell you clearly at startup instead of failing silently.

**The goal is a growing ecosystem.** Bragi ships with a small set of solid adapters. The adapter interface is simple enough that the community can add support for new model families via PRs. Over time, the list of supported models grows — but we don't pretend it's infinite on day one.

---

## Base URL

```
http://localhost:8000/v1
```

All endpoints are prefixed with `/v1` to match OpenAI's API structure.

---

## Authentication

```
Authorization: Bearer <API_KEY>
```

Optional. Configurable via environment variable `BRAGI_API_KEY`. When set, all requests must include this header. When unset, no auth is required.

---

## Model System

### How It Works

1. User sets `HF_TOKEN` once (global, used for all model downloads)
2. User specifies model repo IDs in the config (e.g. `openai/whisper-large-v3`)
3. On startup, Bragi:
   - Downloads the model from Hugging Face (if not already cached)
   - Inspects the model config to detect its architecture type
   - Selects the correct engine adapter
   - Loads the model onto the configured device (CPU/CUDA/MPS)
   - Maps it to the OpenAI model alias for API routing

### What Ships with v1

Bragi launches with adapters for the most widely-used model families. These are production-tested and cover the majority of real-world use cases.

#### STT Adapters (v1)

| Adapter | Models | Features | Why This One |
|---|---|---|---|
| `whisper` | `openai/whisper-*`, `distil-whisper/*` | Transcribe, translate, word timestamps, language detection | Industry standard. Covers 90%+ of STT needs. Huge model variety (tiny → large-v3). |
| `faster-whisper` | `Systran/faster-whisper-*`, `deepdml/faster-whisper-*` | Transcribe, translate, word timestamps, language detection, VAD | CTranslate2-optimized Whisper. Up to 4x faster, lower memory. Drop-in quality match. |
| `nemo` | `nvidia/parakeet-*`, `nvidia/canary-*` | Transcribe, word timestamps | NVIDIA's high-accuracy ASR. Strong on English. ONNX-optimized. |

#### TTS Adapters (v1)

| Adapter | Models | Features | Why This One |
|---|---|---|---|
| `kokoro` | `hexgrad/Kokoro-82M`, `speaches-ai/Kokoro-*-ONNX` | Multi-voice, speed control, streaming | High quality, fast, lightweight, good voice variety. |
| `piper` | `rhasspy/piper-voices-*` | Lightweight, fast, multi-voice | Runs well on CPU, great for low-resource deployments. |

#### What This Means in Practice

With just these adapters, you can:
- Use any Whisper variant for STT (tiny, small, medium, large, distil, turbo — just change the repo name)
- Use faster-whisper for the same Whisper quality at up to 4x the speed with lower memory
- Use Parakeet/NeMo for NVIDIA-optimized high-accuracy English transcription
- Use Kokoro for high-quality TTS or Piper for lightweight/CPU-only TTS
- Swap between any of these by changing the repo name in config

### Future Adapters (Community-Driven)

These are architectures we'd like to support but are not included in v1. Each requires a new adapter (~100-200 lines of Python). Community PRs welcome.

| Adapter | Models | Status |
|---|---|---|
| `ctc` | `facebook/wav2vec2-*`, `facebook/hubert-*` | Planned |
| `coqui` | `coqui/XTTS-v2` | Planned |
| `bark` | `suno/bark`, `suno/bark-small` | Planned |
| `vits` | `facebook/mms-tts-*` | Planned |
| `parler` | `parler-tts/parler-tts-*` | Planned |

### Architecture Detection

For supported adapters, Bragi auto-detects the model type by inspecting the HF repo's `config.json`:

| Detection Signal | Adapter |
|---|---|
| `WhisperForConditionalGeneration` | `whisper` |
| `Systran/faster-whisper` prefix or CTranslate2 model files | `faster-whisper` |
| `EncDecCTCModelBPE`, NeMo ONNX format, `nvidia/parakeet-*` | `nemo` |
| `kokoro` in repo name or config | `kokoro` |
| `.onnx` + piper-schema `config.json` | `piper` |

If a model's architecture doesn't match any installed adapter, Bragi fails at startup with a clear error:

```
Error: Unsupported model architecture 'Wav2Vec2ForCTC' for model 'facebook/wav2vec2-large-960h'.
Installed adapters: whisper, faster-whisper, nemo, kokoro, piper
See https://github.com/yourorg/bragi/blob/main/CONTRIBUTING.md#adding-an-adapter for how to add support.
```

### Adding a New Adapter

Each adapter is a single Python module that implements the STT or TTS protocol. The steps:

1. Create a file in `bragi/adapters/stt/` or `bragi/adapters/tts/`
2. Implement the protocol methods (`transcribe`, `synthesize`, etc.)
3. Register the architecture detection signal
4. Submit a PR

The adapter interface is intentionally minimal so that adding support for a new model family is a self-contained, reviewable contribution — not a major refactor.

---

## Configuration

### Config File (`config.yaml`)

```yaml
hf_token: hf_abc123

server:
  host: 0.0.0.0
  port: 8000
  api_key:
  log_level: info
  max_file_size: 25MB
  workers: 1

device: auto

models:
  whisper-1:
    repo: openai/whisper-large-v3
    device: auto
    compute_type: float16

  tts-1:
    repo: hexgrad/Kokoro-82M
    device: auto

voices:
  alloy:
    speaker_id: af_alloy
  echo:
    speaker_id: am_echo
  nova:
    speaker_id: af_nova
  shimmer:
    speaker_id: af_shimmer
  onyx:
    speaker_id: am_onyx
  fable:
    speaker_id: bf_fable

model_cache_dir: /models
model_ttl: 0
```

### Environment Variables

Environment variables override config file values.

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | Hugging Face token. Used globally for all model downloads. |
| `BRAGI_HOST` | `0.0.0.0` | Server bind address |
| `BRAGI_PORT` | `8000` | Server port |
| `BRAGI_API_KEY` | — | API key for authentication. Unset = no auth. |
| `BRAGI_DEVICE` | `auto` | Compute device: `auto`, `cpu`, `cuda`, `mps` |
| `BRAGI_MODEL_CACHE_DIR` | `/models` | Path to cached model files |
| `BRAGI_LOG_LEVEL` | `info` | Logging level: `debug`, `info`, `warn`, `error` |
| `BRAGI_MAX_FILE_SIZE` | `25MB` | Maximum upload file size |
| `BRAGI_WORKERS` | `1` | Number of Uvicorn workers |
| `BRAGI_MODEL_TTL` | `0` | Seconds before unloading idle models. 0 = never unload. |
| `BRAGI_CONFIG` | `/etc/bragi/config.yaml` | Path to config file |

### Minimal Config (Fastest Start)

```yaml
hf_token: hf_abc123

models:
  whisper-1:
    repo: openai/whisper-large-v3
  tts-1:
    repo: hexgrad/Kokoro-82M
```

Everything else uses sensible defaults. Two lines per model — that's it.

---

## Endpoints

### 1. Create Transcription

```
POST /v1/audio/transcriptions
Content-Type: multipart/form-data
```

Transcribes audio into text in the input language.

#### Request Parameters

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `file` | Yes | file | — | Audio file. Supported formats: `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm` |
| `model` | Yes | string | — | Model alias (e.g. `whisper-1`). Must match a key in the `models` config. |
| `language` | No | string | — | ISO-639-1 language code (e.g. `en`, `fr`, `de`). Improves accuracy and latency. |
| `prompt` | No | string | — | Optional text to guide the model's style or continue a previous segment. |
| `response_format` | No | string | `json` | Output format: `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | No | float | 0 | Sampling temperature, 0.0 to 1.0. |
| `stream` | No | boolean | false | If true, response is streamed via SSE. |
| `timestamp_granularities[]` | No | array | — | `segment` and/or `word`. Only for `verbose_json`. |

#### Response: `json` (default)

```json
{
  "text": "The transcribed text goes here."
}
```

#### Response: `verbose_json`

```json
{
  "task": "transcribe",
  "language": "english",
  "duration": 8.47,
  "text": "The transcribed text goes here.",
  "words": [
    { "word": "The", "start": 0.0, "end": 0.32 },
    { "word": "transcribed", "start": 0.32, "end": 0.88 },
    { "word": "text", "start": 0.88, "end": 1.12 },
    { "word": "goes", "start": 1.12, "end": 1.36 },
    { "word": "here.", "start": 1.36, "end": 1.68 }
  ],
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 8.47,
      "text": "The transcribed text goes here.",
      "tokens": [50364, 440, 1714],
      "temperature": 0.0,
      "avg_logprob": -0.27,
      "compression_ratio": 1.15,
      "no_speech_prob": 0.01
    }
  ]
}
```

#### Response: `text`

```
The transcribed text goes here.
```

#### Response: `srt`

```
1
00:00:00,000 --> 00:00:08,470
The transcribed text goes here.
```

#### Response: `vtt`

```
WEBVTT

00:00:00.000 --> 00:00:08.470
The transcribed text goes here.
```

#### Streaming Response (SSE)

When `stream=true`, the server sends Server-Sent Events:

```
data: {"type":"transcript.text.delta","delta":"The "}

data: {"type":"transcript.text.delta","delta":"transcribed "}

data: {"type":"transcript.text.delta","delta":"text goes here."}

data: {"type":"transcript.text.done","text":"The transcribed text goes here."}
```

---

### 2. Create Speech

```
POST /v1/audio/speech
Content-Type: application/json
```

Generates audio from input text.

#### Request Parameters

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `input` | Yes | string | — | Text to generate audio for. Max 4096 characters. |
| `model` | Yes | string | — | Model alias (e.g. `tts-1`). Must match a key in the `models` config. |
| `voice` | Yes | string or object | — | Voice identifier. Maps to `voices` config. Or custom: `{"id": "speaker_id"}` |
| `instructions` | No | string | — | Voice style control instructions. Max 4096 characters. |
| `response_format` | No | string | `mp3` | Output audio format: `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` |
| `speed` | No | float | 1.0 | Playback speed, 0.25 to 4.0. |

#### Response

Returns raw audio bytes with the appropriate `Content-Type` header:

| Format | Content-Type |
|---|---|
| `mp3` | `audio/mpeg` |
| `opus` | `audio/opus` |
| `aac` | `audio/aac` |
| `flac` | `audio/flac` |
| `wav` | `audio/wav` |
| `pcm` | `audio/pcm` |

#### Streaming Response

When streaming is requested, audio chunks are delivered as:

- **`audio` format**: Raw audio bytes streamed via chunked transfer encoding
- **`sse` format**: Server-Sent Events with base64-encoded audio chunks

```
data: {"type":"audio.delta","delta":"<base64_audio_chunk>"}

data: {"type":"audio.done"}
```

#### Example Request

```bash
curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tts-1",
    "input": "Hello, welcome to Bragi.",
    "voice": "alloy"
  }' \
  --output speech.mp3
```

---

### 3. Create Translation

```
POST /v1/audio/translations
Content-Type: multipart/form-data
```

Translates audio into English.

#### Request Parameters

| Parameter | Required | Type | Default | Description |
|---|---|---|---|---|
| `file` | Yes | file | — | Audio file. Same supported formats as transcription. |
| `model` | Yes | string | — | Model alias. Must support translation (Whisper-family). |
| `prompt` | No | string | — | Optional English text to guide style. |
| `response_format` | No | string | `json` | Output format: `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | No | float | 0 | Sampling temperature, 0.0 to 1.0. |

#### Response: `json` (default)

```json
{
  "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?"
}
```

#### Response: `verbose_json`

```json
{
  "task": "translate",
  "language": "german",
  "duration": 5.23,
  "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 5.23,
      "text": "Hello, my name is Wolfgang and I come from Germany. Where are you heading today?",
      "tokens": [50364, 2425, 11],
      "temperature": 0.0,
      "avg_logprob": -0.31,
      "compression_ratio": 1.08,
      "no_speech_prob": 0.02
    }
  ]
}
```

#### Example Request

```bash
curl http://localhost:8000/v1/audio/translations \
  -H "Content-Type: multipart/form-data" \
  -F file="@german_audio.m4a" \
  -F model="whisper-1"
```

---

### 4. List Models

```
GET /v1/models
```

Returns all models configured in Bragi, their load status, and backend info.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "whisper-1",
      "object": "model",
      "created": 1700000000,
      "owned_by": "bragi",
      "bragi": {
        "repo": "openai/whisper-large-v3",
        "type": "stt",
        "adapter": "whisper",
        "device": "cuda",
        "status": "loaded"
      }
    },
    {
      "id": "tts-1",
      "object": "model",
      "created": 1700000000,
      "owned_by": "bragi",
      "bragi": {
        "repo": "hexgrad/Kokoro-82M",
        "type": "tts",
        "adapter": "kokoro",
        "device": "cuda",
        "status": "loaded"
      }
    }
  ]
}
```

---

## Model Lifecycle

### Startup Flow

```
1. Read config.yaml
2. For each model in config:
   a. Check if model exists in cache dir
   b. If not cached → download from HF Hub using HF_TOKEN
   c. Inspect model config (config.json, model type, file extensions)
   d. Auto-detect architecture → select engine adapter
   e. Load model onto configured device
   f. Register model alias for API routing
3. Start FastAPI server
```

### Lazy Loading (Optional)

When `model_ttl > 0`, models can be loaded on first request instead of startup:

```
1. Request arrives for model "whisper-1"
2. Model not loaded → download if needed → load → process request
3. Model sits idle for TTL seconds → unload from memory
4. Next request → reload
```

### OOM Handling

If a model fails to load on GPU (CUDA OOM):

```
1. Catch OOM error
2. Clear CUDA cache
3. Retry on CPU
4. If CPU also fails → return 503 with clear error message
```

---

## Error Responses

All errors follow the OpenAI error format:

```json
{
  "error": {
    "message": "Invalid file format. Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm",
    "type": "invalid_request_error",
    "param": "file",
    "code": "invalid_file_format"
  }
}
```

### Error Codes

| Code | HTTP | Meaning |
|---|---|---|
| `invalid_file_format` | 400 | Unsupported audio file format |
| `invalid_model` | 400 | Model alias not found in config |
| `invalid_voice` | 400 | Voice ID not found in config |
| `file_too_large` | 413 | File exceeds `max_file_size` |
| `model_not_loaded` | 503 | Model failed to load or is still loading |
| `unsupported_feature` | 400 | Requested feature not supported by this adapter (e.g. translation on a CTC model) |
| `authentication_error` | 401 | Invalid or missing API key |
| `rate_limit_exceeded` | 429 | Too many requests |
| `internal_error` | 500 | Unexpected server error |
| `hf_download_failed` | 503 | Failed to download model from Hugging Face |
| `unsupported_architecture` | 400 | Model architecture not recognized by any adapter |

### HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Bad request — invalid parameters |
| `401` | Unauthorized — invalid or missing API key |
| `413` | Payload too large — file exceeds size limit |
| `422` | Unprocessable entity — valid request but cannot process |
| `429` | Rate limited |
| `500` | Internal server error — backend failure |
| `503` | Service unavailable — model not loaded or download failed |

---

## Supported Audio Formats

### Input (STT / Translation)

`flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, `webm`

### Output (TTS)

`mp3`, `opus`, `aac`, `flac`, `wav`, `pcm`

---

## Docker

### Quick Start

```bash
docker run -p 8000:8000 \
  -e HF_TOKEN=hf_abc123 \
  -v ./config.yaml:/etc/bragi/config.yaml \
  bragi:latest
```

### With GPU

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_abc123 \
  -v ./config.yaml:/etc/bragi/config.yaml \
  bragi:latest
```

### With Persistent Model Cache

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_abc123 \
  -v ./config.yaml:/etc/bragi/config.yaml \
  -v bragi-models:/models \
  bragi:latest
```

### Docker Compose

```yaml
version: "3.8"
services:
  bragi:
    image: bragi:latest
    ports:
      - "8000:8000"
    volumes:
      - ./config.yaml:/etc/bragi/config.yaml
      - bragi-models:/models
    environment:
      - HF_TOKEN=hf_abc123
      - BRAGI_DEVICE=cuda
      - BRAGI_LOG_LEVEL=info
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  bragi-models:
```

---

## Client Compatibility

Bragi is designed as a drop-in replacement. Existing OpenAI client code works by changing only the base URL:

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

transcription = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("audio.mp3", "rb")
)
print(transcription.text)

speech = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello from Bragi."
)
speech.stream_to_file("output.mp3")
```

### curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file="@audio.mp3" \
  -F model="whisper-1"

curl http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"tts-1","input":"Hello","voice":"alloy"}' \
  --output speech.mp3
```

---

## Engine Adapter Interface

Each adapter implements a standard protocol. This is internal — users never see it. Contributors use this to add support for new model families.

### STT Adapter Protocol

```
class STTAdapter:
    def load(model_path, device, **kwargs) → None
    def unload() → None
    def transcribe(audio, language, temperature, word_timestamps) → Transcript
    def translate(audio, temperature) → Transcript
    def get_supported_languages() → list[str]
    def get_sample_rate() → int
    def supports_translation() → bool
    def supports_streaming() → bool
    @staticmethod
    def detect(config_json) → bool
```

### TTS Adapter Protocol

```
class TTSAdapter:
    def load(model_path, device, **kwargs) → None
    def unload() → None
    def synthesize(text, voice, speed, response_format) → audio bytes
    def synthesize_stream(text, voice, speed, response_format) → async iterator[audio bytes]
    def get_available_voices() → list[str]
    def get_sample_rate() → int
    def supports_streaming() → bool
    @staticmethod
    def detect(config_json) → bool
```

### How Detection Works

Each adapter has a static `detect(config_json)` method that returns `True` if it can handle the model. On startup, Bragi iterates through installed adapters and uses the first match. This keeps adapter registration automatic — drop in a new adapter file, and it's available.

### What an Adapter Looks Like (Example)

A typical adapter is a single file, ~100-200 lines:

```
bragi/adapters/stt/whisper.py    (~150 lines)
bragi/adapters/tts/kokoro.py     (~120 lines)
bragi/adapters/tts/piper.py      (~100 lines)
```

The heavy lifting (inference, tokenization, audio processing) is done by the upstream library (faster-whisper, kokoro-onnx, piper-tts). The adapter just bridges that library's API to Bragi's protocol.

---

## Health & Monitoring

```
GET /health
```

```json
{
  "status": "ok",
  "models": {
    "whisper-1": { "status": "loaded", "device": "cuda" },
    "tts-1": { "status": "loaded", "device": "cuda" }
  }
}
```

```
GET /ready
```

Returns 200 only when all configured models are loaded and ready to serve.
