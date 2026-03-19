from __future__ import annotations

import pytest

from app.utils import bare_repo_dir, enforce_error_limit, parse_default_branch


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


class TestBareRepoDir:
    def test_https_url_with_git_suffix(self):
        result = bare_repo_dir("/scratch", "https://github.com/PyPSA/pypsa-eur.git")
        assert result == "/scratch/repos/github.com/PyPSA/pypsa-eur.git"

    def test_https_url_without_git_suffix(self):
        result = bare_repo_dir("/scratch", "https://github.com/PyPSA/pypsa-eur")
        assert result == "/scratch/repos/github.com/PyPSA/pypsa-eur.git"

    def test_both_url_forms_map_to_same_path(self):
        a = bare_repo_dir("/scratch", "https://github.com/PyPSA/pypsa-eur.git")
        b = bare_repo_dir("/scratch", "https://github.com/PyPSA/pypsa-eur")
        assert a == b

    def test_different_host(self):
        result = bare_repo_dir("/scratch", "https://gitlab.example.com/org/repo.git")
        assert result == "/scratch/repos/gitlab.example.com/org/repo.git"

    def test_trailing_slash_stripped(self):
        result = bare_repo_dir("/scratch", "https://github.com/PyPSA/pypsa-eur/")
        assert result == "/scratch/repos/github.com/PyPSA/pypsa-eur.git"


class TestParseDefaultBranch:
    def test_parses_symref_output(self):
        output = "ref: refs/heads/main\tHEAD\nabc123\tHEAD\n"
        assert parse_default_branch(output) == "main"

    def test_parses_non_main_branch(self):
        output = "ref: refs/heads/develop\tHEAD\nabc123\tHEAD\n"
        assert parse_default_branch(output) == "develop"

    def test_falls_back_to_HEAD_without_symref(self):
        output = "abc123\tHEAD\n"
        assert parse_default_branch(output) == "HEAD"

    def test_empty_output_falls_back(self):
        assert parse_default_branch("") == "HEAD"
