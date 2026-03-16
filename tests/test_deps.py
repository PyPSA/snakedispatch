from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.deps import get_store, require_job
from app.store import JobStore


class TestRequireJob:
    def test_returns_record_when_found(self, store):
        record = store.create_job("job-1")
        result = require_job(store, "job-1")
        assert result is record

    def test_raises_404_when_not_found(self, store):
        with pytest.raises(HTTPException) as exc_info:
            require_job(store, "nonexistent-job")
        assert exc_info.value.status_code == 404
        assert "nonexistent-job" in exc_info.value.detail


class TestGetStore:
    def test_returns_store_from_request_state(self):
        mock_store = MagicMock(spec=JobStore)
        request = MagicMock()
        request.app.state.app.store = mock_store
        result = get_store(request)
        assert result is mock_store
