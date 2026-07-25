"""FastAPI dependencies for JWT authentication."""

from fastapi import Header, HTTPException, status
from typing import Any

from services.supabase import verify_token


async def get_current_user(authorization: str = Header(...)) -> dict[str, Any]:
    """Extract and verify the JWT from the Authorization header.

    Expects: Authorization: Bearer <token>
    Returns the user dict from Supabase.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization[7:]  # Strip "Bearer "
    try:
        user = verify_token(token)
        return user
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc