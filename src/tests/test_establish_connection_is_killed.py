"""
Integration test: establish_connection() with is_killed=True must NOT raise
DebridServiceException.

Bug (old code): os.close(_orig_fd) in the finally block → httpx __aexit__ gets
EBADF → propagates to except Exception → raises DebridServiceException.

Expected (fixed code): setsockopt only, no os.close → httpx __aexit__ closes
cleanly → establish_connection exits without exception.

This test directly calls MediaStream.establish_connection() against a real
localhost TCP server so the socket FD path is exercised end-to-end.
"""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import trio
import trio_util

from program.services.streaming.exceptions import (
    DebridServiceClosedConnectionException,
    DebridServiceException,
)
from program.services.streaming.media_stream import MediaStream


async def _serve_one(listener) -> None:
    """Serve a single 206 streaming response."""
    stream = await listener.accept()
    async with stream:
        await stream.receive_some(4096)
        await stream.send_all(
            b"HTTP/1.1 206 Partial Content\r\n"
            b"Content-Range: bytes 0-9999999/10000000\r\n"
            b"Content-Length: 10000000\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b"\r\n"
        )
        with trio.move_on_after(5):
            try:
                while True:
                    await stream.send_all(b"x" * 4096)
                    await trio.sleep(0.001)
            except Exception:
                pass


def _make_media_stream(url: str) -> MediaStream:
    """Bypass MediaStream.__init__ and set only what establish_connection needs."""
    ms = object.__new__(MediaStream)
    ms.is_killed = trio_util.AsyncBool(False)
    ms.is_streaming = trio_util.AsyncBool(False)
    ms.target_url = trio_util.AsyncValue(url)
    ms.provider = "test"
    ms.fh = 1
    ms.async_client = httpx.AsyncClient()
    ms.session_statistics = MagicMock()
    ms.session_statistics.total_session_connections = 0
    ms.file_metadata = MagicMock()
    ms.file_metadata.file_size = 10_000_000
    ms.build_log_message = lambda msg: f"[test] {msg}"

    async def _no_retry(attempt, max_attempts, backoffs):
        return False

    ms._retry_with_backoff = _no_retry
    return ms


@pytest.fixture(autouse=True)
def patch_settings(monkeypatch):
    mock_sm = MagicMock()
    mock_sm.settings.enable_network_tracing = False
    monkeypatch.setattr(
        "program.services.streaming.media_stream.settings_manager",
        mock_sm,
    )


def test_is_killed_does_not_raise_debrid_service_exception():
    """
    When is_killed=True during establish_connection, no DebridServiceException
    should be raised.

    If this test fails, it means EBADF from a double-close is landing in
    `except Exception` and triggering the spurious retry / error-log path.
    """
    raised: list[Exception] = []

    async def run():
        listeners = await trio.open_tcp_listeners(0, host="127.0.0.1")
        port = listeners[0].socket.getsockname()[1]
        ms = _make_media_stream(f"http://127.0.0.1:{port}/")

        async with trio.open_nursery() as nursery:
            nursery.start_soon(_serve_one, listeners[0])

            try:
                async with ms.establish_connection(start=0):
                    # Stream established; signal kill and exit cleanly
                    ms.is_killed.value = True
                    # exiting body → generator resumes → finally block runs
            except Exception as e:
                raised.append(e)
            finally:
                await ms.async_client.aclose()

            nursery.cancel_scope.cancel()

    trio.run(run)

    debrid_errors = [
        e for e in raised
        if isinstance(e, DebridServiceException)
        and not isinstance(e, DebridServiceClosedConnectionException)
    ]
    assert debrid_errors == [], (
        f"DebridServiceException raised: {debrid_errors}\n"
        "This indicates EBADF from os.close(_orig_fd) is reaching except Exception "
        "in the establish_connection retry loop."
    )
