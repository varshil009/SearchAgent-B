"""Authentication routes – proxies to Supabase Auth."""

import json
import logging
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from supabase import create_client, Client

from dependencies import get_current_user

LOGGER = logging.getLogger("research_agent.auth")

router = APIRouter(prefix="/auth", tags=["auth"])


def _get_project_url() -> str:
    """Read the Supabase project URL from supabase_url/url.json."""
    json_path = os.path.join(os.path.dirname(__file__), "..", "supabase_url", "url.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("project_url", "")


def _get_anon_client() -> Client:
    """Create a Supabase client using the anon key (for signup/login)."""
    url = _get_project_url()
    anon_key = os.getenv("SUPABASE_ANON_API", "")
    if not url or not anon_key:
        raise RuntimeError("Project URL and SUPABASE_ANON_API must be configured")
    return create_client(url, anon_key)


class AuthRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(min_length=6)


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict[str, Any]


class UserResponse(BaseModel):
    id: str
    email: str | None
    aud: str | None


@router.post("/signup", response_model=AuthResponse)
async def signup(request: AuthRequest):
    """Sign up a new user via Supabase Auth."""
    try:
        client = _get_anon_client()
        response = client.auth.sign_up(
            {"email": request.email, "password": request.password}
        )
        session = response.session
        user = response.user

        if not session or not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup succeeded but no session returned. Check if email confirmation is required.",
            )

        LOGGER.info("User signed up: %s", user.email)
        return AuthResponse(
            access_token=session.access_token,
            user={
                "id": user.id,
                "email": user.email,
                "aud": user.aud,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Signup failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signup failed: {exc}",
        ) from exc


@router.post("/login", response_model=AuthResponse)
async def login(request: AuthRequest):
    """Log in an existing user via Supabase Auth."""
    try:
        client = _get_anon_client()
        response = client.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
        session = response.session
        user = response.user

        if not session or not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        LOGGER.info("User logged in: %s", user.email)
        return AuthResponse(
            access_token=session.access_token,
            user={
                "id": user.id,
                "email": user.email,
                "aud": user.aud,
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.error("Login failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from exc


@router.get("/me", response_model=UserResponse)
async def get_me(user: dict[str, Any] = Depends(get_current_user)):
    """Get the current authenticated user's info."""
    return UserResponse(
        id=user["id"],
        email=user.get("email"),
        aud=user.get("aud"),
    )
