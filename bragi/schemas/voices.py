from pydantic import BaseModel


class VoiceObject(BaseModel):
    id: str
    name: str
    model: str
    custom: bool
    languages: list[str]


class VoiceListResponse(BaseModel):
    object: str = "list"
    data: list[VoiceObject]


class VoiceCreateResponse(BaseModel):
    id: str
    name: str
    model: str
    created_at: str
