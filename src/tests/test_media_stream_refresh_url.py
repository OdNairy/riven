"""
Regression tests for MediaStream._refresh_download_url.

The coroutine has no internal checkpoint, so calling the blocking DB + provider
unrestrict directly would freeze the whole trio/FUSE event loop for its duration.
The fix runs that work via trio.to_thread.run_sync. These tests assert both that
the blocking call runs off the loop thread and that the loop stays responsive.
"""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import trio
import trio_util

import program.services.streaming.media_stream as ms_mod
from program.services.streaming.media_stream import MediaStream


class _FakeDI:
    """Stand-in for kink's di that returns the same value for any key."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, key):
        return self._value


def _make_stream():
    ms = object.__new__(MediaStream)
    ms.enable_tracing = False
    ms.target_url = trio_util.AsyncValue("old-url")
    ms.file_metadata = SimpleNamespace(original_filename="Show.S01E01.mkv")
    ms.build_log_message = lambda m: m
    return ms


def test_refresh_download_url_runs_off_event_loop_thread(monkeypatch):
    """The blocking DB/provider call must run in a worker thread, not on the loop."""

    loop_thread = threading.get_ident()
    seen: dict[str, int] = {}

    def blocking_get(*_args, **_kwargs):
        seen["thread"] = threading.get_ident()
        return SimpleNamespace(url="fresh-url")

    fake_db = MagicMock()
    fake_db.get_entry_by_original_filename.side_effect = blocking_get
    monkeypatch.setattr(ms_mod, "di", _FakeDI(fake_db))

    ms = _make_stream()

    async def run():
        return await ms._refresh_download_url()

    result = trio.run(run)

    assert result is True
    assert ms.target_url.value == "fresh-url"
    # trio.run drives the loop on the calling thread; the blocking call must NOT
    # have executed there (it would, if _refresh_download_url skipped to_thread).
    assert seen["thread"] != loop_thread


def test_event_loop_stays_responsive_during_refresh(monkeypatch):
    """While the blocking refresh runs, other trio tasks must keep progressing."""

    block_seconds = 0.3

    def blocking_get(*_args, **_kwargs):
        time.sleep(block_seconds)  # simulate blocking SQLAlchemy + provider unrestrict
        return SimpleNamespace(url="fresh-url")

    fake_db = MagicMock()
    fake_db.get_entry_by_original_filename.side_effect = blocking_get
    monkeypatch.setattr(ms_mod, "di", _FakeDI(fake_db))

    ms = _make_stream()

    async def run():
        ticks = 0
        done = trio.Event()

        async def ticker():
            nonlocal ticks
            while not done.is_set():
                await trio.sleep(0.01)
                ticks += 1

        async with trio.open_nursery() as nursery:
            nursery.start_soon(ticker)
            result = await ms._refresh_download_url()
            done.set()

        return result, ticks

    result, ticks = trio.run(run)

    assert result is True
    # If the loop were blocked for the whole window, ticks would be ~0.
    assert ticks >= 5, f"event loop appears blocked during refresh (ticks={ticks})"
