"""
Regression test: SO_LINGER EBADF double-close in establish_connection.

Bug: os.close(_orig_fd) in the finally-block (is_killed path) causes httpx's
__aexit__ to raise OSError(EBADF). In establish_connection this lands in
`except Exception` and triggers spurious retry logging (4x RST instead of 1x).

Two tests document the two behaviors:
  test_double_close_raises_ebadf   – old code path: double-close → EBADF propagates
  test_setsockopt_only_no_ebadf    – proper fix path: setsockopt only → clean exit
"""
import errno
import os
import socket as _socket_module
import struct

import httpx
import pytest
import trio


async def _serve_one(listener) -> None:
    """Accept one connection and keep it open with streaming data."""
    stream = await listener.accept()
    async with stream:
        await stream.receive_some(4096)
        await stream.send_all(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 10000000\r\n"
            b"Content-Type: application/octet-stream\r\n"
            b"\r\n"
        )
        with trio.move_on_after(5):
            try:
                while True:
                    await stream.send_all(b"x" * 4096)
                    await trio.sleep(0.01)
            except Exception:
                pass


def _get_orig_fd(response: httpx.Response) -> int:
    ns = response.extensions.get("network_stream")
    if ns is None:
        return -1
    raw = ns.get_extra_info("socket")
    if raw is None:
        return -1
    return raw.fileno() if hasattr(raw, "fileno") else -1


def test_double_close_raises_ebadf():
    """
    os.close(orig_fd) inside the stream body causes httpx __aexit__ to raise
    OSError(EBADF). This is the root cause of spurious retries in establish_connection.
    """
    raised: list[OSError] = []

    async def run():
        listeners = await trio.open_tcp_listeners(0, host="127.0.0.1")
        port = listeners[0].socket.getsockname()[1]

        async with trio.open_nursery() as nursery:
            nursery.start_soon(_serve_one, listeners[0])

            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream("GET", f"http://127.0.0.1:{port}/") as resp:
                        orig_fd = _get_orig_fd(resp)
                        assert orig_fd >= 0, "httpx did not expose socket FD"

                        # Simulate establish_connection finally-block (old code)
                        os.close(orig_fd)

                        # httpx __aexit__ will now try to close the same FD → EBADF
                except OSError as e:
                    raised.append(e)

            nursery.cancel_scope.cancel()

    trio.run(run)

    assert len(raised) == 1, f"Expected EBADF from httpx __aexit__, got: {raised}"
    assert raised[0].errno == errno.EBADF, f"Expected EBADF (errno 9), got errno {raised[0].errno}"


def test_setsockopt_only_no_ebadf():
    """
    Setting SO_LINGER via dup_fd without closing orig_fd lets httpx close the
    socket cleanly — no EBADF, no spurious exception, RST still sent by kernel.
    This is the proper fix.
    """
    raised: list[Exception] = []

    async def run():
        listeners = await trio.open_tcp_listeners(0, host="127.0.0.1")
        port = listeners[0].socket.getsockname()[1]

        async with trio.open_nursery() as nursery:
            nursery.start_soon(_serve_one, listeners[0])

            async with httpx.AsyncClient() as client:
                try:
                    async with client.stream("GET", f"http://127.0.0.1:{port}/") as resp:
                        orig_fd = _get_orig_fd(resp)
                        assert orig_fd >= 0, "httpx did not expose socket FD"

                        # Proper fix: set SO_LINGER via dup, leave orig_fd for httpx
                        dup_fd = os.dup(orig_fd)
                        with _socket_module.socket(fileno=dup_fd) as s:
                            s.setsockopt(
                                _socket_module.SOL_SOCKET,
                                _socket_module.SO_LINGER,
                                struct.pack("ii", 1, 0),
                            )
                        # orig_fd still open; httpx __aexit__ closes it with RST

                except OSError as e:
                    raised.append(e)
                except Exception as e:
                    raised.append(e)

            nursery.cancel_scope.cancel()

    trio.run(run)

    assert raised == [], f"Expected clean exit with proper fix, got: {raised}"
