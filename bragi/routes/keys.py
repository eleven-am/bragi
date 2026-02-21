from fastapi import APIRouter, Request

from bragi.schemas.errors import KeyNotFoundError
from bragi.schemas.keys import (
    KeyCreateRequest,
    KeyCreateResponse,
    KeyListResponse,
    KeyObject,
)

router = APIRouter()


@router.post("/admin/keys")
async def create_key(request: Request, body: KeyCreateRequest) -> KeyCreateResponse:
    key_store = request.app.state.key_store
    stored, raw_key = await key_store.create(body.name)
    return KeyCreateResponse(
        id=stored.id,
        name=stored.name,
        key=raw_key,
        created_at=stored.created_at,
    )


@router.get("/admin/keys")
async def list_keys(request: Request) -> KeyListResponse:
    key_store = request.app.state.key_store
    keys = await key_store.list_all()
    return KeyListResponse(
        data=[
            KeyObject(
                id=k.id,
                name=k.name,
                prefix=k.prefix,
                created_at=k.created_at,
                last_used_at=k.last_used_at,
                is_active=k.is_active,
            )
            for k in keys
        ]
    )


@router.delete("/admin/keys/{key_id}")
async def delete_key(request: Request, key_id: str):
    key_store = request.app.state.key_store
    deleted = await key_store.delete(key_id)
    if not deleted:
        raise KeyNotFoundError(key_id)
    return {"deleted": True, "id": key_id}
