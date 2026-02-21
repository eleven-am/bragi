from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

_KEY_PREFIX = "br-"
_TOKEN_BYTES = 32
_DISPLAY_PREFIX_LEN = 8


def _generate_raw_key() -> str:
    return _KEY_PREFIX + secrets.token_hex(_TOKEN_BYTES)


def _hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


@dataclass
class StoredKey:
    id: str
    name: str
    key_hash: str
    prefix: str
    created_at: str
    last_used_at: str | None
    is_active: bool


class KeyStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS keys (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                prefix TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT,
                is_active INTEGER DEFAULT 1
            )"""
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def create(self, name: str) -> tuple[StoredKey, str]:
        raw_key = _generate_raw_key()
        key_hash = _hash_key(raw_key)
        key_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        prefix = raw_key[:len(_KEY_PREFIX) + _DISPLAY_PREFIX_LEN]

        assert self._db is not None
        await self._db.execute(
            "INSERT INTO keys (id, name, key_hash, prefix, created_at, last_used_at, is_active) "
            "VALUES (?, ?, ?, ?, ?, NULL, 1)",
            (key_id, name, key_hash, prefix, created_at),
        )
        await self._db.commit()

        stored = StoredKey(
            id=key_id,
            name=name,
            key_hash=key_hash,
            prefix=prefix,
            created_at=created_at,
            last_used_at=None,
            is_active=True,
        )
        return stored, raw_key

    async def validate(self, raw_key: str) -> StoredKey | None:
        key_hash = _hash_key(raw_key)
        assert self._db is not None
        async with self._db.execute(
            "SELECT * FROM keys WHERE key_hash = ? AND is_active = 1", (key_hash,)
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            r = dict(row)
            r["is_active"] = bool(r["is_active"])
            return StoredKey(**r)

    async def get_by_id(self, key_id: str) -> StoredKey | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM keys WHERE id = ?", (key_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            r = dict(row)
            r["is_active"] = bool(r["is_active"])
            return StoredKey(**r)

    async def list_all(self) -> list[StoredKey]:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM keys ORDER BY created_at") as cursor:
            rows = await cursor.fetchall()
            result = []
            for row in rows:
                r = dict(row)
                r["is_active"] = bool(r["is_active"])
                result.append(StoredKey(**r))
            return result

    async def delete(self, key_id: str) -> bool:
        existing = await self.get_by_id(key_id)
        if existing is None:
            return False

        assert self._db is not None
        await self._db.execute("DELETE FROM keys WHERE id = ?", (key_id,))
        await self._db.commit()
        return True

    async def update_last_used(self, key_id: str) -> None:
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            "UPDATE keys SET last_used_at = ? WHERE id = ?", (now, key_id)
        )
        await self._db.commit()

    async def is_empty(self) -> bool:
        assert self._db is not None
        async with self._db.execute("SELECT COUNT(*) FROM keys") as cursor:
            row = await cursor.fetchone()
            return row[0] == 0
