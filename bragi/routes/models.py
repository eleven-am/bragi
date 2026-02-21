import time

from fastapi import APIRouter, Request

from bragi.schemas.responses import ModelListResponse, ModelObject

router = APIRouter()


@router.get("/models")
async def list_models(request: Request):
    registry = request.app.state.registry
    models = registry.list_models()

    data = [
        ModelObject(
            id=m.alias,
            created=int(time.time()),
            owned_by="bragi",
        )
        for m in models
    ]

    return ModelListResponse(data=data)
