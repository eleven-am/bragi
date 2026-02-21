from pydantic import BaseModel


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class BragiError(Exception):
    def __init__(
        self,
        message: str,
        status_code: int = 400,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        self.param = param
        self.code = code

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorDetail(
                message=self.message,
                type=self.error_type,
                param=self.param,
                code=self.code,
            )
        )


class InvalidModelError(BragiError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' not found. Check your config.",
            status_code=400,
            error_type="invalid_request_error",
            param="model",
            code="invalid_model",
        )


class InvalidVoiceError(BragiError):
    def __init__(self, voice: str):
        super().__init__(
            message=f"Voice '{voice}' not found. Check your config.",
            status_code=400,
            error_type="invalid_request_error",
            param="voice",
            code="invalid_voice",
        )


class InvalidFileFormatError(BragiError):
    def __init__(self, fmt: str | None = None):
        msg = "Invalid file format."
        if fmt:
            msg = f"Invalid file format: {fmt}."
        msg += " Supported formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm"
        super().__init__(
            message=msg,
            status_code=400,
            error_type="invalid_request_error",
            param="file",
            code="invalid_file_format",
        )


class FileTooLargeError(BragiError):
    def __init__(self, max_size: str):
        super().__init__(
            message=f"File exceeds maximum size of {max_size}.",
            status_code=413,
            error_type="invalid_request_error",
            param="file",
            code="file_too_large",
        )


class ModelNotLoadedError(BragiError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' is not loaded or still loading.",
            status_code=503,
            error_type="server_error",
            param="model",
            code="model_not_loaded",
        )


class UnsupportedFeatureError(BragiError):
    def __init__(self, feature: str, model: str):
        super().__init__(
            message=f"Model '{model}' does not support {feature}.",
            status_code=400,
            error_type="invalid_request_error",
            param="model",
            code="unsupported_feature",
        )


class VoiceConflictError(BragiError):
    def __init__(self, name: str):
        super().__init__(
            message=f"Voice '{name}' already exists.",
            status_code=409,
            error_type="invalid_request_error",
            param="name",
            code="voice_conflict",
        )


class VoiceCloningNotSupportedError(BragiError):
    def __init__(self, model: str):
        super().__init__(
            message=f"Model '{model}' does not support voice cloning.",
            status_code=400,
            error_type="invalid_request_error",
            param="model",
            code="voice_cloning_not_supported",
        )


class KeyNotFoundError(BragiError):
    def __init__(self, key_id: str):
        super().__init__(
            message=f"API key '{key_id}' not found.",
            status_code=404,
            error_type="invalid_request_error",
            param="key_id",
            code="key_not_found",
        )


class AuthenticationError(BragiError):
    def __init__(self):
        super().__init__(
            message="Invalid or missing API key.",
            status_code=401,
            error_type="authentication_error",
            code="authentication_error",
        )
