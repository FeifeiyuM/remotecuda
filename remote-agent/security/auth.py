"""Bearer-token authentication dependency."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status


class BearerAuth:
    def __init__(self, token: str, *, enabled: bool = True) -> None:
        self._enabled = enabled
        self._token = token.encode("utf-8") if enabled else b""

    async def __call__(self, authorization: str | None = Header(default=None)) -> None:
        if not self._enabled:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        supplied = authorization[len("Bearer ") :].encode("utf-8")
        if not hmac.compare_digest(supplied, self._token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
