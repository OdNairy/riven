"""
Tests for the TorBox downloader.

These exercise the response-parsing and request-shaping logic that was verified
against the TorBox OpenAPI spec (https://api.torbox.app/openapi.json):

  - checkcached files carry NO ``id`` field, so the requestdl ``file_id`` is the
    file's 0-based position in the list;
  - ControlTorrent types ``torrent_id`` as an integer;
  - user/me uses ``is_subscribed`` / ``total_downloaded``;
  - createtorrent may return ``torrent_id`` or ``queued_id``;
  - requestdl resolves via a 3xx redirect, and a permanently gone link must map to
    DebridServiceLinkUnavailable.

The downloader is instantiated via ``__new__`` to bypass the network calls in
``__init__``/``validate``; the ``api.session`` is a MagicMock in every test.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from program.services.downloaders.models import TorrentContainer, UnrestrictedLink
from program.services.downloaders.torbox import (
    TorBoxAPI,
    TorBoxDownloader,
    TorBoxError,
    _requestdl_permalink,
)
from program.services.streaming.exceptions.debrid_service_exception import (
    DebridServiceLinkUnavailable,
)
from program.utils.request import CircuitBreakerOpen

API_KEY = "TESTKEY123"

# Sizes chosen to clear the default filesize filters (movie >= 700MB, episode >= 100MB).
GB = 1_000_000_000


class FakeResp:
    """Minimal stand-in for SmartResponse used by the downloader methods."""

    def __init__(self, status_code=200, json_data=None, headers=None, reason="OK"):
        self.status_code = status_code
        self._json = json_data
        self.headers = headers or {}
        self.reason = reason
        self.closed = False

    @property
    def ok(self):
        return 200 <= self.status_code < 400

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json

    def close(self):
        self.closed = True


def make_downloader(api_key=API_KEY):
    dl = TorBoxDownloader.__new__(TorBoxDownloader)
    dl.key = "torbox"
    dl.settings = SimpleNamespace(api_key=api_key, enabled=True)
    dl.api = SimpleNamespace(
        session=MagicMock(),
        api_key=api_key,
        BASE_URL=TorBoxAPI.BASE_URL,
    )
    return dl


def envelope(data, success=True):
    return {"success": success, "data": data}


# ---------------------------------------------------------------------------
# _ordered_torbox_files — the core file_id fix
# ---------------------------------------------------------------------------


def test_ordered_files_uses_position_when_no_id():
    # checkcached files have no "id"; file_id must be the 0-based position.
    raw = [
        {"name": "a.mkv", "size": 1},
        {"name": "b.mkv", "size": 2},
        {"name": "c.mkv", "size": 3},
    ]
    result = TorBoxDownloader._ordered_torbox_files(raw)
    assert [fid for fid, _ in result] == [0, 1, 2]
    assert [f["name"] for _, f in result] == ["a.mkv", "b.mkv", "c.mkv"]


def test_ordered_files_prefers_explicit_id():
    # mylist files carry an "id"; it must be honoured over the position.
    raw = [
        {"id": 5, "name": "a.mkv"},
        {"id": 9, "name": "b.mkv"},
    ]
    result = TorBoxDownloader._ordered_torbox_files(raw)
    assert [fid for fid, _ in result] == [5, 9]


def test_ordered_files_lowercases_keys_and_skips_junk():
    raw = [{"Name": "A.mkv", "Size": 1}, "not-a-dict", {"name": "b.mkv"}]
    result = TorBoxDownloader._ordered_torbox_files(raw)
    assert [(fid, f.get("name")) for fid, f in result] == [(0, "A.mkv"), (1, "b.mkv")]


def test_ordered_files_flattens_nested_payload():
    nested = {"files": [{"id": 0, "name": "n1.mkv"}, {"id": 1, "name": "n2.mkv"}]}
    result = TorBoxDownloader._ordered_torbox_files(nested)
    assert [f["name"] for _, f in result] == ["n1.mkv", "n2.mkv"]


def test_ordered_files_empty():
    assert TorBoxDownloader._ordered_torbox_files([]) == []
    assert TorBoxDownloader._ordered_torbox_files(None) == []


# ---------------------------------------------------------------------------
# _build_container_from_checkcached — regression for "always uncached"
# ---------------------------------------------------------------------------


def test_build_container_assigns_positional_file_id():
    dl = make_downloader()
    # A non-video file precedes the real video file, so the video file's index
    # (and therefore its requestdl file_id) is 1, not 0.
    row = {
        "name": "Movie.2020.1080p",
        "hash": "deadbeef",
        "files": [
            {"name": "Movie.2020.1080p/readme.txt", "size": 1024},
            {"name": "Movie.2020.1080p/Movie.2020.1080p.mkv", "size": 3 * GB},
        ],
    }
    container = dl._build_container_from_checkcached("deadbeef", "movie", row)
    assert isinstance(container, TorrentContainer)
    assert len(container.files) == 1
    f = container.files[0]
    assert f.file_id == 1  # positional index survives filtering of earlier files
    assert f.filename == "Movie.2020.1080p.mkv"
    assert f.filesize == 3 * GB
    # download_url is deferred to prepare_download()
    assert f.download_url is None


def test_build_container_skips_samples_and_nonvideo():
    dl = make_downloader()
    row = {
        "files": [
            {"name": "sample.mkv", "size": 3 * GB},  # 'sample' -> skipped
            {"name": "notes.txt", "size": 3 * GB},  # non-video -> skipped
            {"name": "real.movie.mkv", "size": 3 * GB},
        ]
    }
    container = dl._build_container_from_checkcached("h", "movie", row)
    assert container is not None
    assert [f.filename for f in container.files] == ["real.movie.mkv"]
    assert container.files[0].file_id == 2


def test_build_container_prefers_short_name_for_filename():
    dl = make_downloader()
    row = {
        "files": [
            {
                "name": "Movie.2020.1080p/Movie.2020.1080p.mkv",
                "short_name": "Movie.2020.1080p.mkv",
                "size": 3 * GB,
            }
        ]
    }
    container = dl._build_container_from_checkcached("h", "movie", row)
    assert container is not None
    assert container.files[0].filename == "Movie.2020.1080p.mkv"


def test_build_container_returns_none_when_no_video_files():
    dl = make_downloader()
    row = {"files": [{"name": "readme.txt", "size": 10}]}
    assert dl._build_container_from_checkcached("h", "movie", row) is None


# ---------------------------------------------------------------------------
# get_instant_availability — checkcached hit -> prepare_download
# ---------------------------------------------------------------------------


def test_get_instant_availability_sets_torrent_id_and_download_urls():
    dl = make_downloader()
    infohash = "abc123"

    checkcached_row = {
        "name": "Ep",
        "hash": infohash,
        "files": [
            {"name": "junk.nfo", "size": 500},
            {"name": "Show.S01E01.1080p.mkv", "size": 500_000_000},
        ],
    }

    def get_router(url, **_kwargs):
        assert "checkcached" in url
        return FakeResp(json_data=envelope([checkcached_row]))

    def post_router(url, **_kwargs):
        assert "createtorrent" in url
        return FakeResp(json_data=envelope({"torrent_id": 4242, "hash": infohash}))

    dl.api.session.get = MagicMock(side_effect=get_router)
    dl.api.session.post = MagicMock(side_effect=post_router)

    container = dl.get_instant_availability(infohash, "episode")

    assert container is not None
    assert container.torrent_id == "4242"
    assert len(container.files) == 1

    df = container.files[0]
    assert df.file_id == 1  # positional index in checkcached list
    expected_url = _requestdl_permalink(API_KEY, "4242", 1)
    assert df.download_url == expected_url

    # torrent_info wired up for the orchestrator's active_stream
    assert container.torrent_info is not None
    assert container.torrent_info.id == "4242"
    assert 1 in container.torrent_info.files


def test_get_instant_availability_returns_none_when_not_cached():
    dl = make_downloader()
    dl.api.session.get = MagicMock(return_value=FakeResp(json_data=envelope([])))
    assert dl.get_instant_availability("nope", "movie") is None


def test_get_instant_availability_propagates_circuit_breaker():
    dl = make_downloader()
    dl.api.session.get = MagicMock(side_effect=CircuitBreakerOpen("api.torbox.app"))
    with pytest.raises(CircuitBreakerOpen):
        dl.get_instant_availability("x", "movie")


# ---------------------------------------------------------------------------
# add_torrent — torrent_id / queued_id parsing
# ---------------------------------------------------------------------------


def test_add_torrent_parses_torrent_id():
    dl = make_downloader()
    dl.api.session.post = MagicMock(
        return_value=FakeResp(json_data=envelope({"torrent_id": 777}))
    )
    assert dl.add_torrent("hash") == "777"


def test_add_torrent_parses_queued_id():
    dl = make_downloader()
    dl.api.session.post = MagicMock(
        return_value=FakeResp(json_data=envelope({"queued_id": 888}))
    )
    assert dl.add_torrent("hash") == "888"


def test_add_torrent_raises_without_id():
    dl = make_downloader()
    dl.api.session.post = MagicMock(
        return_value=FakeResp(json_data=envelope({"hash": "h"}))
    )
    with pytest.raises(TorBoxError):
        dl.add_torrent("hash")


# ---------------------------------------------------------------------------
# delete_torrent — ControlTorrent wants an integer torrent_id
# ---------------------------------------------------------------------------


def test_delete_torrent_sends_integer_id():
    dl = make_downloader()
    dl.api.session.post = MagicMock(return_value=FakeResp(status_code=204))
    dl.delete_torrent("12345")

    _, kwargs = dl.api.session.post.call_args
    assert kwargs["json"] == {"torrent_id": 12345, "operation": "delete"}
    assert isinstance(kwargs["json"]["torrent_id"], int)


def test_delete_torrent_keeps_non_numeric_id():
    dl = make_downloader()
    dl.api.session.post = MagicMock(return_value=FakeResp(status_code=204))
    dl.delete_torrent("abc")
    _, kwargs = dl.api.session.post.call_args
    assert kwargs["json"]["torrent_id"] == "abc"


# ---------------------------------------------------------------------------
# get_user_info — is_subscribed / plan / total_downloaded
# ---------------------------------------------------------------------------


def test_user_info_premium_via_subscription_and_total_downloaded():
    dl = make_downloader()
    dl.api.session.get = MagicMock(
        return_value=FakeResp(
            json_data=envelope(
                {
                    "id": 1,
                    "email": "u@example.com",
                    "plan": 2,
                    "is_subscribed": True,
                    "total_downloaded": 123456,
                }
            )
        )
    )
    info = dl.get_user_info()
    assert info is not None
    assert info.premium_status == "premium"
    assert info.total_downloaded_bytes == 123456


def test_user_info_free_plan():
    dl = make_downloader()
    dl.api.session.get = MagicMock(
        return_value=FakeResp(
            json_data=envelope({"id": 1, "plan": 0, "is_subscribed": False})
        )
    )
    info = dl.get_user_info()
    assert info is not None
    assert info.premium_status == "free"


# ---------------------------------------------------------------------------
# _row_to_torrent_info — mylist parsing (files carry an "id")
# ---------------------------------------------------------------------------


def test_row_to_torrent_info_uses_mylist_file_ids():
    dl = make_downloader()
    row = {
        "id": 55,
        "name": "Show/Show.S01E01.mkv",
        "hash": "hh",
        "download_state": "cached",
        "size": 500_000_000,
        "files": [
            {
                "id": 0,
                "name": "Show/Show.S01E01.mkv",
                "short_name": "Show.S01E01.mkv",
                "size": 500_000_000,
            },
        ],
    }
    info = dl._row_to_torrent_info(row)
    assert info.id == "55"
    assert 0 in info.files
    tf = info.files[0]
    assert tf.download_url == _requestdl_permalink(API_KEY, 55, 0)


# ---------------------------------------------------------------------------
# unrestrict_link — redirect handling, gone links, backoff, no body read
# ---------------------------------------------------------------------------


def test_unrestrict_link_follows_redirect_and_uses_stream():
    dl = make_downloader()
    cdn = "https://cdn.torbox.app/dl/abc/movie.mkv?x=1"
    resp = FakeResp(status_code=302, headers={"Location": cdn})
    dl.api.session.get = MagicMock(return_value=resp)

    result = dl.unrestrict_link("https://api.torbox.app/v1/api/torrents/requestdl?...")

    assert isinstance(result, UnrestrictedLink)
    assert result.download == cdn
    assert result.filename == "movie.mkv"

    # Must not read the body: stream=True + allow_redirects=False, and closed after.
    _, kwargs = dl.api.session.get.call_args
    assert kwargs["stream"] is True
    assert kwargs["allow_redirects"] is False
    assert resp.closed is True


def test_unrestrict_link_gone_raises_link_unavailable():
    dl = make_downloader()
    resp = FakeResp(status_code=404)
    dl.api.session.get = MagicMock(return_value=resp)
    with pytest.raises(DebridServiceLinkUnavailable):
        dl.unrestrict_link("https://api.torbox.app/v1/api/torrents/requestdl?...")
    assert resp.closed is True


def test_unrestrict_link_rate_limited_propagates_circuit_breaker():
    dl = make_downloader()
    resp = FakeResp(status_code=429)
    dl.api.session.get = MagicMock(return_value=resp)
    with pytest.raises(CircuitBreakerOpen):
        dl.unrestrict_link("https://api.torbox.app/v1/api/torrents/requestdl?...")
    assert resp.closed is True


def test_unrestrict_link_unexpected_status_returns_none():
    dl = make_downloader()
    resp = FakeResp(status_code=200)  # 200 with no redirect: not what we expect
    dl.api.session.get = MagicMock(return_value=resp)
    assert (
        dl.unrestrict_link("https://api.torbox.app/v1/api/torrents/requestdl") is None
    )
    assert resp.closed is True
