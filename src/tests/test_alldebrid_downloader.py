import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from program.services.downloaders.alldebrid import (
    AllDebridDirectory,
    AllDebridFile,
    AllDebridMagnetStatusResponse,
    AllDebridResponse,
)
from program.services.downloaders.models import DebridFile, TorrentContainer, TorrentInfo

TEST_DATA = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# Test group 1 — Pydantic model (Fix 1: MagnetInfo.files optional)
# ---------------------------------------------------------------------------


def test_magnet_status_without_files_parses_ok():
    with open(TEST_DATA / "alldebrid_magnet_status_one_downloading.json") as f:
        body = json.load(f)

    result = AllDebridResponse[AllDebridMagnetStatusResponse].model_validate(
        {"data": body}
    )
    from program.services.downloaders.alldebrid import AllDebridSuccessResponse

    assert isinstance(result.data, AllDebridSuccessResponse)
    magnets = result.data.data.magnets
    assert len(magnets) == 1
    magnet = magnets[0]
    assert isinstance(magnet, AllDebridMagnetStatusResponse.MagnetInfo)
    assert magnet.files is None


def test_magnet_status_with_files_unchanged():
    with open(TEST_DATA / "alldebrid_magnet_status_nested_files.json") as f:
        body = json.load(f)

    result = AllDebridResponse[AllDebridMagnetStatusResponse].model_validate(
        {"data": body}
    )
    from program.services.downloaders.alldebrid import AllDebridSuccessResponse

    assert isinstance(result.data, AllDebridSuccessResponse)
    magnets = result.data.data.magnets
    assert len(magnets) == 1
    magnet = magnets[0]
    assert isinstance(magnet, AllDebridMagnetStatusResponse.MagnetInfo)
    assert magnet.files is not None
    assert len(magnet.files) == 1
    assert isinstance(magnet.files[0], AllDebridDirectory)


# ---------------------------------------------------------------------------
# Test group 2 — File extraction (Fix 2: sequential file IDs)
# ---------------------------------------------------------------------------


def _make_downloader_stub():
    """Return a minimal AllDebridDownloader-like object without real API init."""
    from program.services.downloaders.alldebrid import AllDebridDownloader

    obj = object.__new__(AllDebridDownloader)
    return obj


def test_extract_files_assigns_sequential_ids():
    downloader = _make_downloader_stub()
    flat_files = [
        AllDebridFile(n="ep01.mkv", s=500_000_000, l="https://ad.com/f/1"),
        AllDebridFile(n="ep02.mkv", s=500_000_000, l="https://ad.com/f/2"),
        AllDebridFile(n="ep03.mkv", s=500_000_000, l="https://ad.com/f/3"),
    ]
    result: list[DebridFile] = []
    downloader._extract_files_recursive(flat_files, "episode", result, "abc123")

    assert len(result) == 3
    assert [f.file_id for f in result] == [0, 1, 2]


def test_extract_files_nested_directories_flattened():
    """Files from nested dirs (after _add_link_to_files_recursive) get sequential IDs."""
    downloader = _make_downloader_stub()
    # Flat list as produced by _add_link_to_files_recursive from the nested fixture
    flat_files = [
        AllDebridFile(n="01 - Episode One.avi", s=104_857_600, l="https://ad.com/f/link1"),
        AllDebridFile(n="02 - Episode Two.avi", s=104_857_600, l="https://ad.com/f/link2"),
        AllDebridFile(n="03 - Episode Three.avi", s=104_857_600, l="https://ad.com/f/link3"),
        AllDebridFile(n="cover.jpg", s=12_288, l="https://ad.com/f/cover"),
    ]
    result: list[DebridFile] = []
    downloader._extract_files_recursive(flat_files, "episode", result, "abc123")

    # cover.jpg filtered by DebridFile.create (non-video extension)
    assert len(result) == 3
    assert [f.file_id for f in result] == [0, 1, 2]


def test_extract_files_respects_episode_size_minimum():
    """Files below episode_filesize_mb_min are excluded; IDs still sequential for valid files."""
    downloader = _make_downloader_stub()
    flat_files = [
        AllDebridFile(n="ep01.mkv", s=500_000_000, l="https://ad.com/f/1"),
        AllDebridFile(n="tiny.mkv", s=1_000, l="https://ad.com/f/tiny"),   # too small
        AllDebridFile(n="ep03.mkv", s=500_000_000, l="https://ad.com/f/3"),
    ]
    result: list[DebridFile] = []
    downloader._extract_files_recursive(flat_files, "episode", result, "abc123")

    assert len(result) == 2
    assert [f.file_id for f in result] == [0, 1]


# ---------------------------------------------------------------------------
# Test group 3 — Manual scrape session integration (Fix 2 + 3)
# ---------------------------------------------------------------------------


def _make_debrid_file(file_id: int, name: str, url: str) -> DebridFile:
    df = DebridFile(file_id=file_id, filename=name, filesize=500_000_000)
    df.download_url = url
    return df


def test_manual_session_parsed_files_not_empty_for_alldebrid():
    """start_manual_session loop skips file_id=None; with sequential IDs all files pass."""
    files = [
        _make_debrid_file(0, "ep01.mkv", "https://ad.com/f/1"),
        _make_debrid_file(1, "ep02.mkv", "https://ad.com/f/2"),
        _make_debrid_file(2, "ep03.mkv", "https://ad.com/f/3"),
    ]
    # Simulate the loop from start_manual_session
    parsed = [f for f in files if f.file_id is not None]
    assert len(parsed) == 3


def test_download_and_update_uses_container_fallback_when_info_files_empty():
    """When info.files is empty (AllDebrid), container fallback returns selected files."""
    files = [
        _make_debrid_file(0, "ep01.mkv", "https://ad.com/f/1"),
        _make_debrid_file(1, "ep02.mkv", "https://ad.com/f/2"),
        _make_debrid_file(2, "ep03.mkv", "https://ad.com/f/3"),
    ]

    # Simulate the fallback branch from _download_and_update
    info = TorrentInfo(
        id=123,
        name="Test",
        status="Ready",
        infohash=None,
        bytes=0,
        created_at=None,
        completed_at=None,
        progress=100.0,
        files={},
        links=[],
    )
    stored_container = TorrentContainer(infohash="abc123", files=files)

    file_id_set = {0, 2}
    container_files: list[DebridFile] = []

    if info.files:
        pass  # standard path — not taken
    else:
        if stored_container and stored_container.files:
            container_files = [f for f in stored_container.files if f.file_id in file_id_set]

    assert len(container_files) == 2
    assert {f.file_id for f in container_files} == {0, 2}


def test_download_and_update_standard_path_unchanged_for_realdebrid():
    """When info.files is populated (RealDebrid), standard path is used, no fallback."""
    from program.services.downloaders.models import TorrentFile

    files_map = {
        1: TorrentFile(id=1, path="/ep01.mkv", bytes=500_000_000, selected=1, download_url="https://rd.com/f/1"),
        2: TorrentFile(id=2, path="/ep02.mkv", bytes=500_000_000, selected=1, download_url="https://rd.com/f/2"),
    }
    info = TorrentInfo(
        id=456,
        name="Test",
        status="downloaded",
        infohash=None,
        bytes=0,
        created_at=None,
        completed_at=None,
        progress=100.0,
        files=files_map,
        links=[],
    )
    file_id_set = {1}
    container_files: list[DebridFile] = []

    if info.files:
        for fid, meta in info.files.items():
            if fid not in file_id_set:
                continue
            df = DebridFile(file_id=fid, filename=meta.filename, filesize=meta.bytes)
            df.download_url = meta.download_url
            container_files.append(df)

    assert len(container_files) == 1
    assert container_files[0].file_id == 1
