from pydantic import BaseModel


class KeyCreateRequest(BaseModel):
    name: str


class KeyObject(BaseModel):
    id: str
    name: str
    prefix: str
    created_at: str
    last_used_at: str | None = None
    is_active: bool


class KeyCreateResponse(BaseModel):
    id: str
    name: str
    key: str
    created_at: str


class KeyListResponse(BaseModel):
    object: str = "list"
    data: list[KeyObject]
