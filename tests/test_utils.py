from __future__ import annotations

import pytest

from app.utils import enforce_error_limit


class TestEnforceErrorLimit:
    def test_logs_warning_below_threshold(self, caplog):
        import logging

        exc = ValueError("disk error")
        with caplog.at_level(logging.WARNING, logger="app.utils"):
            enforce_error_limit(5, "test context", exc, threshold=10)
        assert "test context" in caplog.text

    def test_raises_at_threshold(self):
        exc = OSError("connection refused")
        with pytest.raises(RuntimeError, match="10 consecutive"):
            enforce_error_limit(10, "Job abc123", exc, threshold=10)

    def test_raises_above_threshold(self):
        exc = RuntimeError("timeout")
        with pytest.raises(RuntimeError, match="15 consecutive"):
            enforce_error_limit(15, "GC loop", exc, threshold=10)

    def test_custom_label(self):
        exc = ConnectionError("ssh error")
        with pytest.raises(RuntimeError, match="SSH errors"):
            enforce_error_limit(5, "Job xyz", exc, threshold=5, label="SSH errors")

    def test_custom_threshold(self, caplog):
        import logging

        exc = ValueError("transient")
        with caplog.at_level(logging.WARNING, logger="app.utils"):
            enforce_error_limit(3, "sync loop", exc, threshold=5)
        assert "sync loop" in caplog.text

    def test_log_message_includes_count_and_threshold(self, caplog):
        import logging

        exc = ValueError("oops")
        with caplog.at_level(logging.WARNING, logger="app.utils"):
            enforce_error_limit(2, "monitor", exc, threshold=5, label="I/O errors")
        assert "2/5" in caplog.text
        assert "I/O errors" in caplog.text

    def test_exception_chained(self):
        exc = OSError("original")
        with pytest.raises(RuntimeError) as exc_info:
            enforce_error_limit(10, "ctx", exc, threshold=10)
        assert exc_info.value.__cause__ is exc
