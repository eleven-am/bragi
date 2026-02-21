# Bragi

Self-hosted, OpenAI-compatible STT/TTS server. Drop-in replacement for the OpenAI Audio API — transcriptions, translations, text-to-speech — running on your own hardware.

## What this was

The idea was to build an "Ollama for voice." One Docker image you pull, point at a config file listing which models you want, and it downloads them from HuggingFace at startup. No cloud API keys, no vendor lock-in, just local inference with a familiar REST API.

It supports multiple STT and TTS backends through an adapter system:

- **STT:** faster-whisper, Vosk, FunASR (Paraformer), Moonshine, SpeechBrain, NVIDIA Parakeet
- **TTS:** Kokoro, Piper, Coqui XTTS, F5-TTS, Fish Speech, Qwen3-TTS

Each adapter wraps a different open-source model behind a common interface. You configure which models to load, Bragi figures out which adapter handles them, and exposes everything through OpenAI-compatible endpoints.

## Why I stopped

The Docker image came out to 16.9 GB.

Python ML libraries (PyTorch, CUDA bindings, adapter dependencies) are heavy. PyTorch alone ships ~5-6 GB of NVIDIA libraries via pip, on top of whatever CUDA runtime the base image already has. Baking every adapter into one image means pulling in nemo-toolkit, speechbrain, spacy, funasr, and all their transitive dependencies.

Ollama solves this by being a compiled Go binary that shells out to llama.cpp (C++). The actual CUDA libraries come from the host via the NVIDIA Container Toolkit — nothing is bundled in the image. There's no Python, no pip, no framework overhead.

There's no equivalent approach for the Python voice/audio ML ecosystem. You could slim it down — use system CUDA instead of pip CUDA, install adapters at runtime instead of bake-time — but you're still looking at a multi-gigabyte image just for PyTorch + one or two adapters. The "pull and it just works" experience I was going for doesn't really work when the pull is 17 GB.

## What's here

The code works. If you don't mind the image size, you can build and run it:

```
cd bragi
docker build -t bragi .
docker run --gpus all -v bragi_data:/data -p 8000:8000 bragi
```

The server starts, downloads models on first run, and serves the OpenAI Audio API at `localhost:8000`. API keys are auto-generated on first boot (check the logs).

### Endpoints

- `POST /v1/audio/transcriptions` — speech-to-text
- `POST /v1/audio/translations` — speech translation
- `POST /v1/audio/speech` — text-to-speech
- `GET /v1/models` — list loaded models
- `GET /health` — health check

### Config

Either a YAML file at `/etc/bragi/config.yaml` or environment variables:

```yaml
device: auto
models:
  whisper:
    repo: Systran/faster-whisper-large-v3
  kokoro:
    repo: hexgrad/Kokoro-82M
```

```
BRAGI_DEVICE=cuda
BRAGI_PORT=8000
BRAGI_MODEL_CACHE_DIR=/data/models
```

## If you want to continue this

The adapter system is the interesting part. Each adapter implements a `detect(cfg)` class method and a `load(repo, device)` instance method. Adding a new backend is one file. The rest is standard FastAPI.

The hard problem is the image size. If someone figures out a way to ship Python ML inference in a small container — or rewrites the adapter layer in something compiled — that would make the whole thing viable.

## License

MIT
