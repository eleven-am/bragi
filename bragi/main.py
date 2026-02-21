import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bragi.adapters.faster_whisper import FasterWhisperAdapter
from bragi.adapters.kokoro import KokoroAdapter
from bragi.adapters.stt import STTAdapter

_optional_adapters: list[type] = []

try:
    from bragi.adapters.vosk_adapter import VoskAdapter
    _optional_adapters.append(VoskAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.paraformer import ParaformerAdapter
    _optional_adapters.append(ParaformerAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.moonshine import MoonshineAdapter
    _optional_adapters.append(MoonshineAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.speechbrain_adapter import SpeechBrainAdapter
    _optional_adapters.append(SpeechBrainAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.parakeet import ParakeetAdapter
    _optional_adapters.append(ParakeetAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.piper import PiperAdapter
    _optional_adapters.append(PiperAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.coqui_xtts import CoquiXTTSAdapter
    _optional_adapters.append(CoquiXTTSAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.f5_tts import F5TTSAdapter
    _optional_adapters.append(F5TTSAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.fish_speech import FishSpeechAdapter
    _optional_adapters.append(FishSpeechAdapter)
except ImportError:
    pass

try:
    from bragi.adapters.qwen3_tts import Qwen3TTSAdapter
    _optional_adapters.append(Qwen3TTSAdapter)
except ImportError:
    pass

from bragi.config import load_config
from bragi.keys.store import KeyStore
from bragi.middleware.auth import AuthMiddleware
from bragi.registry import ModelInfo, ModelRegistry
from bragi.routes import keys, models, speech, transcriptions, translations, voices
from bragi.schemas.errors import BragiError
from bragi.voices.store import VoiceStore

logger = logging.getLogger("bragi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config()

    logging.basicConfig(
        level=getattr(logging, config.server.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    registry = ModelRegistry()

    voice_base = Path(config.voice_store_dir) if config.voice_store_dir else Path(config.model_cache_dir) / "voices"
    voice_store = VoiceStore(
        db_path=voice_base / "voices.db",
        audio_dir=voice_base / "audio",
    )
    await voice_store.initialize()

    key_base = Path(config.key_store_dir) if config.key_store_dir else Path(config.model_cache_dir) / "keys"
    key_store = KeyStore(db_path=key_base / "keys.db")
    await key_store.initialize()

    if await key_store.is_empty():
        stored, raw_key = await key_store.create("default")
        logger.info("Generated API key: %s", raw_key)

    adapter_classes: list[type] = [FasterWhisperAdapter, KokoroAdapter] + _optional_adapters

    for alias, model_config in config.models.items():
        cfg = {"repo": model_config.repo}

        matched = None
        for cls in adapter_classes:
            if cls.detect(cfg):
                matched = cls
                break

        if matched is None:
            logger.warning("No adapter for model '%s' (repo: %s)", alias, model_config.repo)
            continue

        adapter = matched()
        device = model_config.device if model_config.device != "auto" else config.device
        adapter.load(model_config.repo, device, compute_type=model_config.compute_type)

        info = ModelInfo(
            alias=alias,
            model_type="stt" if isinstance(adapter, STTAdapter) else "tts",
            repo=model_config.repo,
            device=device,
            status="loaded",
        )

        if isinstance(adapter, STTAdapter):
            registry.register_stt(alias, adapter, info)
        else:
            registry.register_tts(alias, adapter, info)

        logger.info("Loaded model '%s' (%s) on %s", alias, model_config.repo, device)

    for cv in await voice_store.list_all():
        if cv.adapter_alias:
            registry.register_custom_voice(cv.name, cv.adapter_alias)

    app.state.config = config
    app.state.registry = registry
    app.state.voice_store = voice_store
    app.state.key_store = key_store

    logger.info("Bragi started on %s:%d", config.server.host, config.server.port)

    yield

    await key_store.close()
    await voice_store.close()
    registry.unload_all()
    logger.info("Bragi shutdown complete")


def create_app() -> FastAPI:
    application = FastAPI(title="Bragi", version="0.1.0", lifespan=lifespan)

    application.add_middleware(AuthMiddleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(transcriptions.router, prefix="/v1")
    application.include_router(speech.router, prefix="/v1")
    application.include_router(translations.router, prefix="/v1")
    application.include_router(models.router, prefix="/v1")
    application.include_router(voices.router, prefix="/v1")
    application.include_router(keys.router, prefix="/v1")

    @application.exception_handler(BragiError)
    async def bragi_error_handler(request: Request, exc: BragiError):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_response().model_dump(),
        )

    @application.get("/health")
    async def health(request: Request):
        registry: ModelRegistry = request.app.state.registry
        model_status = {}
        for info in registry.list_models():
            model_status[info.alias] = {
                "status": info.status,
                "device": info.device,
            }
        return {"status": "ok", "models": model_status}

    @application.get("/ready")
    async def ready(request: Request):
        return {"status": "ok"}

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    config = load_config()
    uvicorn.run(
        "bragi.main:app",
        host=config.server.host,
        port=config.server.port,
        workers=config.server.workers,
    )
