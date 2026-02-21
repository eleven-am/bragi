from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite


@dataclass
class CustomVoice:
    id: str
    name: str
    transcript: str
    original_filename: str
    adapter_alias: str
    created_at: str


class VoiceStore:
    def __init__(self, db_path: Path, audio_dir: Path) -> None:
        self._db_path = db_path
        self._audio_dir = audio_dir
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._audio_dir.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute(
            """CREATE TABLE IF NOT EXISTS voices (
                id TEXT PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                transcript TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                adapter_alias TEXT NOT NULL,
                created_at TEXT NOT NULL
            )"""
        )
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def create(
        self,
        name: str,
        transcript: str,
        audio_data: bytes,
        original_filename: str,
        adapter_alias: str,
    ) -> CustomVoice:
        voice_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()

        voice_dir = self._audio_dir / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        (voice_dir / "reference.wav").write_bytes(audio_data)

        assert self._db is not None
        await self._db.execute(
            "INSERT INTO voices (id, name, transcript, original_filename, adapter_alias, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (voice_id, name, transcript, original_filename, adapter_alias, created_at),
        )
        await self._db.commit()

        return CustomVoice(
            id=voice_id,
            name=name,
            transcript=transcript,
            original_filename=original_filename,
            adapter_alias=adapter_alias,
            created_at=created_at,
        )

    async def get_by_name(self, name: str) -> CustomVoice | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM voices WHERE name = ?", (name,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return CustomVoice(**dict(row))

    async def get_by_id(self, voice_id: str) -> CustomVoice | None:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM voices WHERE id = ?", (voice_id,)) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return CustomVoice(**dict(row))

    async def list_all(self) -> list[CustomVoice]:
        assert self._db is not None
        async with self._db.execute("SELECT * FROM voices ORDER BY created_at") as cursor:
            rows = await cursor.fetchall()
            return [CustomVoice(**dict(row)) for row in rows]

    async def delete(self, voice_id: str) -> bool:
        voice = await self.get_by_id(voice_id)
        if voice is None:
            return False

        assert self._db is not None
        await self._db.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
        await self._db.commit()

        voice_dir = self._audio_dir / voice_id
        ref = voice_dir / "reference.wav"
        if ref.exists():
            ref.unlink()
        if voice_dir.exists():
            voice_dir.rmdir()

        return True

    def get_reference_audio(self, voice_id: str) -> bytes:
        ref = self._audio_dir / voice_id / "reference.wav"
        return ref.read_bytes()
