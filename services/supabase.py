"""Supabase service for thread & message management + JWT verification."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.request import urlopen

from supabase import create_client, Client

LOGGER = logging.getLogger("research_agent.supabase")

_SUPABASE_CLIENT: Client | None = None
_JWKS_CACHE: dict[str, Any] | None = None


def _get_project_url() -> str:
    """Read the Supabase project URL from supabase_url/url.json."""
    json_path = os.path.join(os.path.dirname(__file__), "..", "supabase_url", "url.json")
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        url = data.get("project_url", "")
        if not url:
            raise RuntimeError("project_url is empty in supabase_url/url.json")
        return url
    except FileNotFoundError:
        raise RuntimeError(
            "supabase_url/url.json not found. Create it with: {\"project_url\": \"https://your-project.supabase.co\"}"
        )
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in supabase_url/url.json: {e}")


def get_supabase_admin() -> Client:
    """Get or create the Supabase admin client using the service_role key."""
    global _SUPABASE_CLIENT
    if _SUPABASE_CLIENT is None:
        url = _get_project_url()
        key = os.getenv("SUPABASE_SERVICE_ROLE_API", "")
        if not url or not key:
            raise RuntimeError(
                "Project URL and SUPABASE_SERVICE_ROLE_API must be configured"
            )
        _SUPABASE_CLIENT = create_client(url, key)
    return _SUPABASE_CLIENT


def _get_jwks() -> list[dict[str, Any]]:
    """Fetch JWKS from Supabase Auth endpoint."""
    global _JWKS_CACHE
    if _JWKS_CACHE is None:
        url = _get_project_url()
        jwks_url = f"{url.rstrip('/')}/auth/v1/.well-known/jwks.json"
        try:
            with urlopen(jwks_url) as response:
                _JWKS_CACHE = json.loads(response.read())
        except Exception as exc:
            LOGGER.error("Failed to fetch JWKS: %s", exc)
            raise RuntimeError(f"Failed to fetch JWKS: {exc}") from exc
    return _JWKS_CACHE.get("keys", [])


def verify_token(token: str) -> dict[str, Any]:
    """Verify a JWT token using Supabase Auth API (server-side).

    Uses the service_role client's auth.get_user() which calls the
    Supabase Auth REST API to verify the token server-side.
    This is more reliable than local JWT verification.

    Returns the user dict with keys: id, email, etc.
    Raises RuntimeError if token is invalid.
    """
    supabase = get_supabase_admin()
    try:
        # Use the admin client's auth API to verify the token server-side
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if user is None:
            raise RuntimeError("Token verification returned no user")
        return {
            "id": user.id,
            "email": user.email,
            "aud": user.aud,
        }
    except Exception as exc:
        LOGGER.error("Token verification failed: %s", exc)
        # Fallback: try using the REST API directly
        try:
            return _verify_token_rest(token)
        except Exception:
            raise RuntimeError(f"Invalid or expired token: {exc}") from exc


def _verify_token_rest(token: str) -> dict[str, Any]:
    """Fallback: verify JWT by calling Supabase Auth REST API directly."""
    import httpx

    url = _get_project_url()
    api_key = os.getenv("SUPABASE_SERVICE_ROLE_API", "")
    headers = {
        "apikey": api_key,
        "Authorization": f"Bearer {token}",
    }
    response = httpx.get(
        f"{url.rstrip('/')}/auth/v1/user",
        headers=headers,
        timeout=10,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Token verification failed: {response.status_code} {response.text}"
        )
    data = response.json()
    return {
        "id": data["id"],
        "email": data.get("email"),
        "aud": data.get("aud"),
    }


def create_thread(user_id: str, title: str = "New Chat") -> dict[str, Any]:
    """Create a new thread for the given user."""
    supabase = get_supabase_admin()
    thread_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "id": thread_id,
        "user_id": user_id,
        "title": title,
        "summary": None,
        "created_at": now,
        "updated_at": now,
    }
    result = supabase.table("threads").insert(data).execute()
    if not result.data:
        raise RuntimeError("Failed to create thread")
    return result.data[0]


def list_threads(user_id: str) -> list[dict[str, Any]]:
    """List all threads for a user, ordered by updated_at DESC."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("threads")
        .select("*")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .execute()
    )
    return result.data or []


def get_thread(thread_id: str, user_id: str) -> dict[str, Any] | None:
    """Get a single thread by id, verifying ownership."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("threads")
        .select("*")
        .eq("id", thread_id)
        .eq("user_id", user_id)
        .execute()
    )
    if result.data:
        return result.data[0]
    return None


def get_messages(thread_id: str, user_id: str) -> list[dict[str, Any]]:
    """Get all messages for a thread, ordered by sequence_no ASC.

    Verifies the thread belongs to the user first.
    """
    thread = get_thread(thread_id, user_id)
    if thread is None:
        raise RuntimeError("Thread not found or access denied")

    supabase = get_supabase_admin()
    result = (
        supabase.table("messages")
        .select("*")
        .eq("thread_id", thread_id)
        .order("sequence_no", desc=False)
        .execute()
    )
    return result.data or []


def get_next_sequence_no(thread_id: str) -> int:
    """Get the next sequence number for a thread."""
    supabase = get_supabase_admin()
    result = (
        supabase.table("messages")
        .select("sequence_no")
        .eq("thread_id", thread_id)
        .order("sequence_no", desc=True)
        .limit(1)
        .execute()
    )
    if result.data:
        return result.data[0]["sequence_no"] + 1
    return 1


def add_message(
    thread_id: str,
    role: str,
    content: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Add a message to a thread with auto-incrementing sequence_no."""
    supabase = get_supabase_admin()
    sequence_no = get_next_sequence_no(thread_id)
    message_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    data = {
        "id": message_id,
        "thread_id": thread_id,
        "sequence_no": sequence_no,
        "role": role,
        "content": content,
        "metadata": metadata or {},
        "created_at": now,
    }
    result = supabase.table("messages").insert(data).execute()
    if not result.data:
        raise RuntimeError("Failed to add message")
    return result.data[0]


def update_thread_timestamp(thread_id: str) -> None:
    """Update the updated_at timestamp of a thread."""
    supabase = get_supabase_admin()
    now = datetime.now(timezone.utc).isoformat()
    supabase.table("threads").update({"updated_at": now}).eq(
        "id", thread_id
    ).execute()


def update_thread_title(thread_id: str, title: str) -> None:
    """Update the title of a thread."""
    supabase = get_supabase_admin()
    now = datetime.now(timezone.utc).isoformat()
    supabase.table("threads").update({"title": title, "updated_at": now}).eq(
        "id", thread_id
    ).execute()