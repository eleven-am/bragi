import asyncio

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from bragi.schemas.errors import AuthenticationError


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ("/health", "/ready"):
            return await call_next(request)

        if not request.url.path.startswith("/v1"):
            return await call_next(request)

        key_store = getattr(request.app.state, "key_store", None)
        if key_store is None:
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            error = AuthenticationError()
            return JSONResponse(
                status_code=error.status_code,
                content=error.to_response().model_dump(),
            )

        token = auth_header[7:]
        stored_key = await key_store.validate(token)
        if stored_key is None:
            error = AuthenticationError()
            return JSONResponse(
                status_code=error.status_code,
                content=error.to_response().model_dump(),
            )

        asyncio.create_task(key_store.update_last_used(stored_key.id))

        return await call_next(request)
