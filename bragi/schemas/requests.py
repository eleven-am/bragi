from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    input: str = Field(..., max_length=4096)
    model: str | None = None
    voice: str
    instructions: str | None = Field(None, max_length=4096)
    response_format: str = "mp3"
    speed: float = Field(1.0, ge=0.25, le=4.0)
