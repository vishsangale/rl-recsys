from pathlib import Path
import pytest
from rl_recsys.data.download import _md5, download_file


def test_md5_returns_32_char_hex(tmp_path):
    f = tmp_path / "file.txt"
    f.write_bytes(b"hello")
    result = _md5(f)
    assert isinstance(result, str) and len(result) == 32


def test_download_file_skips_existing_without_network(tmp_path, monkeypatch):
    dest = tmp_path / "existing.txt"
    dest.write_bytes(b"data")
    called = []
    monkeypatch.setattr("requests.get", lambda *a, **kw: called.append(True))
    download_file("http://unused.invalid", dest)
    assert called == []


def test_download_file_raises_on_bad_checksum_after_download(tmp_path, monkeypatch):
    dest = tmp_path / "file.txt"

    class FakeResp:
        headers = {"content-length": "5"}
        def raise_for_status(self): pass
        def iter_content(self, size): yield b"hello"

    monkeypatch.setattr("requests.get", lambda *a, **kw: FakeResp())
    with pytest.raises(ValueError, match="Checksum mismatch"):
        download_file("http://unused.invalid", dest,
                      expected_md5="deadbeef00000000deadbeef00000000")
