"""
Regression tests for MediaStream.establish_connection's handling of 5xx
upstream errors.

A 5xx from the debrid CDN can mean the specific edge node handed out for this
URL is degraded, while a fresh unrestrict may route to a healthy node. On the
first 5xx, establish_connection should try refreshing the download URL (the
same recovery already used for 404/410/503 and connection errors) before
exhausting retries and raising.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import trio
import trio_util

import program.services.streaming.media_stream as ms_mod
from program.services.streaming.exceptions import DebridServiceException
from program.services.streaming.media_stream import MediaStream


def _error_response(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://example.invalid/file")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(
        f"{status_code} error", request=request, response=response
    )


class _FailingStreamCM:
    """Async context manager whose __aenter__ raises like a bad HTTP response."""

    def __init__(self, status_code: int) -> None:
        self._status_code = status_code

    async def __aenter__(self):
        raise _error_response(self._status_code)

    async def __aexit__(self, *_exc_info) -> bool:
        return False


class _OkStream:
    """Minimal stand-in for the httpx streaming response on the happy path."""

    status_code = 206
    headers: dict = {}
    extensions: dict = {}

    def raise_for_status(self) -> None:
        return None


class _OkStreamCM:
    def __init__(self, stream: "_OkStream") -> None:
        self._stream = stream

    async def __aenter__(self) -> "_OkStream":
        return self._stream

    async def __aexit__(self, *_exc_info) -> bool:
        return False


class _ScriptedAsyncClient:
    """Fake async_client.stream() that plays back a scripted sequence of results."""

    def __init__(self, results: list[int | None]) -> None:
        # Each entry is a status code to fail with, or None to succeed.
        self._results = list(results)
        self.call_count = 0

    def stream(self, *, method: str, url: str, headers=None, extensions=None):
        result = self._results[min(self.call_count, len(self._results) - 1)]
        self.call_count += 1

        if result is None:
            return _OkStreamCM(_OkStream())

        return _FailingStreamCM(result)


def _make_stream(async_client: _ScriptedAsyncClient) -> MediaStream:
    ms = object.__new__(MediaStream)
    ms.provider = "torbox"
    ms.async_client = async_client
    ms.target_url = trio_util.AsyncValue("http://example.invalid/old")
    ms.file_metadata = SimpleNamespace(
        original_filename="Show.S01E01.mkv", file_size=1000
    )
    ms.session_statistics = SimpleNamespace(total_session_connections=0)
    ms.is_killed = trio_util.AsyncBool(False)
    ms.enable_tracing = False
    ms.build_log_message = lambda m: m
    return ms


@pytest.fixture(autouse=True)
def _no_network_tracing(monkeypatch):
    monkeypatch.setattr(
        ms_mod.settings_manager,
        "settings",
        SimpleNamespace(enable_network_tracing=False),
    )


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch):
    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(ms_mod.trio, "sleep", _no_sleep)


def test_5xx_triggers_url_refresh_before_giving_up(monkeypatch):
    """A persistent 5xx should attempt exactly one URL refresh, then raise."""

    async_client = _ScriptedAsyncClient([502, 502, 502, 502])
    ms = _make_stream(async_client)

    refresh_mock = AsyncMock(return_value=False)
    ms._refresh_download_url = refresh_mock

    async def run():
        async with ms.establish_connection(0, end=9):
            pass

    with pytest.raises(DebridServiceException):
        trio.run(run)

    assert refresh_mock.await_count == 1
    assert async_client.call_count == 4  # max_attempts


def test_5xx_recovers_after_successful_refresh(monkeypatch):
    """If the refreshed URL works, the connection should succeed without raising."""

    async_client = _ScriptedAsyncClient([502, None])
    ms = _make_stream(async_client)

    refresh_mock = AsyncMock(return_value=True)
    ms._refresh_download_url = refresh_mock

    async def run():
        async with ms.establish_connection(0, end=9) as stream:
            return stream.status_code

    status_code = trio.run(run)

    assert status_code == 206
    assert refresh_mock.await_count == 1
    assert async_client.call_count == 2


def test_5xx_only_refreshes_on_first_attempt(monkeypatch):
    """Refresh should not be retried on every subsequent 5xx in the same call."""

    async_client = _ScriptedAsyncClient([502, 502, 502, 502])
    ms = _make_stream(async_client)

    refresh_mock = AsyncMock(return_value=True)
    ms._refresh_download_url = refresh_mock

    async def run():
        async with ms.establish_connection(0, end=9):
            pass

    with pytest.raises(DebridServiceException):
        trio.run(run)

    # Even though refresh "succeeded" and returned a URL, subsequent attempts
    # still fail (e.g. every node is degraded) - refresh must not be retried
    # on every attempt, only on the first.
    assert refresh_mock.await_count == 1
