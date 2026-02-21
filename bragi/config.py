from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    pass


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    max_file_size: str = "25MB"
    workers: int = 1


class ModelConfig(BaseModel):
    repo: str
    device: str = "auto"
    compute_type: str | None = None


class BragiConfig(BaseModel):
    hf_token: str | None = None
    server: ServerConfig = ServerConfig()
    device: str = "auto"
    models: dict[str, ModelConfig] = {}
    model_cache_dir: str = "/models"
    model_ttl: int = 0
    voice_store_dir: str | None = None
    key_store_dir: str | None = None


_SIZE_UNITS: dict[str, int] = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 ** 2,
    "GB": 1024 ** 3,
    "TB": 1024 ** 4,
}

_SIZE_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(B|KB|MB|GB|TB)\s*$", re.IGNORECASE)


def parse_file_size(size_str: str) -> int:
    match = _SIZE_PATTERN.match(size_str)
    if not match:
        raise ValueError(f"Invalid file size format: {size_str!r}")
    value = float(match.group(1))
    unit = match.group(2).upper()
    return int(value * _SIZE_UNITS[unit])


def load_config() -> BragiConfig:
    config_path = Path(os.environ.get("BRAGI_CONFIG", "/etc/bragi/config.yaml"))

    data: dict = {}
    if config_path.is_file():
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}

    env_overrides: dict[str, tuple[list[str], type]] = {
        "HF_TOKEN": (["hf_token"], str),
        "BRAGI_HOST": (["server", "host"], str),
        "BRAGI_PORT": (["server", "port"], int),
        "BRAGI_DEVICE": (["device"], str),
        "BRAGI_MODEL_CACHE_DIR": (["model_cache_dir"], str),
        "BRAGI_LOG_LEVEL": (["server", "log_level"], str),
        "BRAGI_MAX_FILE_SIZE": (["server", "max_file_size"], str),
        "BRAGI_WORKERS": (["server", "workers"], int),
        "BRAGI_MODEL_TTL": (["model_ttl"], int),
        "BRAGI_VOICE_STORE_DIR": (["voice_store_dir"], str),
        "BRAGI_KEY_STORE_DIR": (["key_store_dir"], str),
    }

    for env_var, (key_path, cast) in env_overrides.items():
        value = os.environ.get(env_var)
        if value is None:
            continue

        target = data
        for key in key_path[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[key_path[-1]] = cast(value)

    return BragiConfig(**data)
