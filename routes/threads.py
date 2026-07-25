"""Thread management routes."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from dependencies import get_current_user
from services.supabase import (
    create_thread,
    list_threads,
    get_thread,
    get_messages,
    update_thread_title,
)

LOGGER = logging.getLogger("research_agent.threads")

router = APIRouter(prefix="/threads", tags=["threads"])


class CreateThreadRequest(BaseModel):
    title: str = Field(default="New Chat", max_length=255)


class ThreadResponse(BaseModel):
    id: str
    user_id: str
    title: str | None
    summary: str | None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: str
    thread_id: str
    sequence_no: int
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: str


@router.get("", response_model=list[ThreadResponse])
async def list_user_threads(user: dict[str, Any] = Depends(get_current_user)):
    """List all threads for the current user."""
    threads = list_threads(user["id"])
    return [ThreadResponse(**t) for t in threads]


@router.post("", response_model=ThreadResponse, status_code=status.HTTP_201_CREATED)
async def create_new_thread(
    request: CreateThreadRequest,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Create a new thread for the current user."""
    thread = create_thread(user["id"], request.title)
    return ThreadResponse(**thread)


@router.get("/{thread_id}", response_model=ThreadResponse)
async def get_thread_by_id(
    thread_id: str,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Get a single thread by ID."""
    thread = get_thread(thread_id, user["id"])
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    return ThreadResponse(**thread)


@router.get("/{thread_id}/messages", response_model=list[MessageResponse])
async def get_thread_messages(
    thread_id: str,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Get all messages for a thread."""
    try:
        messages = get_messages(thread_id, user["id"])
        return [MessageResponse(**m) for m in messages]
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.patch("/{thread_id}/title", response_model=ThreadResponse)
async def update_thread_title_endpoint(
    thread_id: str,
    request: CreateThreadRequest,
    user: dict[str, Any] = Depends(get_current_user),
):
    """Update the title of a thread."""
    thread = get_thread(thread_id, user["id"])
    if thread is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Thread not found",
        )
    update_thread_title(thread_id, request.title)
    updated = get_thread(thread_id, user["id"])
    return ThreadResponse(**updated)
