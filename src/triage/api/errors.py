"""Stable, language-neutral error codes. The Angular client translates them."""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(_request: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(
            status_code=error.status_code, content={"code": error.code, "message": error.message}
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, error: RequestValidationError) -> JSONResponse:
        fields = sorted(
            {".".join(str(part) for part in item["loc"][1:]) for item in error.errors()}
        )
        return JSONResponse(
            status_code=422,
            content={"code": "VALIDATION_FAILED", "message": "Invalid request", "fields": fields},
        )
