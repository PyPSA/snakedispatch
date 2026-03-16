from __future__ import annotations

import logging
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Response

from app.backends.base import ComputeBackend
from app.config import Settings
from app.deps import HealthCache, get_backend, get_settings, provide_health_cache
from app.models import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_health_response(backend_ok: bool) -> HealthResponse:
    status: Literal["ok", "degraded"] = "ok" if backend_ok else "degraded"
    return HealthResponse(status=status, backend=backend_ok)


@router.get("/health", response_model=HealthResponse)
async def health_check(
    backend: Annotated[ComputeBackend, Depends(get_backend)],
    response: Response,
) -> HealthResponse:
    """Check service backend health."""
    backend_ok = await backend.check_connectivity()
    if not backend_ok:
        logger.warning("Health check: backend unreachable")
        response.status_code = 503
    return _build_health_response(backend_ok)


@router.get("/health/cached", response_model=HealthResponse)
async def health_check_cached(
    backend: Annotated[ComputeBackend, Depends(get_backend)],
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
    cache: Annotated[HealthCache, Depends(provide_health_cache)],
) -> HealthResponse:
    """Check service backend health, using cached result within TTL."""
    now = time.monotonic()
    if (
        cache["backend_ok"] is not None
        and now - cache["checked_at"] < settings.HEALTH_CACHE_TTL_SECONDS
    ):
        backend_ok = cache["backend_ok"]
    else:
        backend_ok = await backend.check_connectivity()
        if not backend_ok:
            logger.warning("Health check (cached): backend unreachable")
        cache["backend_ok"] = backend_ok
        cache["checked_at"] = now
    if not backend_ok:
        response.status_code = 503
    return _build_health_response(backend_ok)
