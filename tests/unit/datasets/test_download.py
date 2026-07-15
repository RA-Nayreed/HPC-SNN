import gzip
import hashlib

import pytest
from conftest import write_event_h5

from fedapfa.datasets import download as module
from fedapfa.datasets.validation import DatasetExpectation


def _payloads(tmp_path):
    source = write_event_h5(tmp_path / "source.h5", (0, 1))
    compressed = gzip.compress(source.read_bytes(), mtime=0)
    files = {name: compressed for name in module.DATASET_FILES["shd"]}
    manifest = "\n".join(f"{hashlib.md5(data).hexdigest()}  {name}" for name, data in files.items()).encode()
    return files, manifest


def test_mocked_atomic_download_and_validated_extract(tmp_path, monkeypatch):
    files, manifest = _payloads(tmp_path)
    monkeypatch.setitem(module.OFFICIAL_EXPECTATIONS, "shd_train.h5", DatasetExpectation(2, 2))
    monkeypatch.setitem(module.OFFICIAL_EXPECTATIONS, "shd_test.h5", DatasetExpectation(2, 2))

    def downloader(url, destination):
        destination.write_bytes(manifest if url.endswith("md5sums.txt") else files[url.rsplit("/", 1)[-1]])

    outputs = module.download_dataset("shd", tmp_path / "raw", downloader)
    assert all(path.is_file() for path in outputs)
    assert not list((tmp_path / "raw" / "shd").glob("*.part"))


def test_interrupted_download_removes_part(tmp_path):
    def downloader(url, destination):
        if url.endswith("md5sums.txt"):
            destination.write_text("0" * 32 + "  shd_train.h5.gz\n" + "0" * 32 + "  shd_test.h5.gz\n")
        else:
            destination.write_bytes(b"partial")
            raise OSError("interrupted")

    with pytest.raises(module.DownloadError, match="interrupted"):
        module.download_dataset("shd", tmp_path, downloader)
    assert not list((tmp_path / "shd").glob("*.part"))


def test_checksum_parser_and_missing_checksum(tmp_path):
    assert module.parse_checksums("abc bad") == {}

    def downloader(url, destination):
        destination.write_text("")

    with pytest.raises(module.DownloadError, match="no entry"):
        module.download_dataset("shd", tmp_path, downloader)


def test_checksum_mismatch_removes_invalid_archive(tmp_path):
    files, _ = _payloads(tmp_path)
    manifest = ("0" * 32 + "  shd_train.h5.gz\n" + "0" * 32 + "  shd_test.h5.gz\n").encode()

    def downloader(url, destination):
        destination.write_bytes(manifest if url.endswith("md5sums.txt") else files[url.rsplit("/", 1)[-1]])

    with pytest.raises(module.DownloadError, match="MD5 mismatch"):
        module.download_dataset("shd", tmp_path / "raw", downloader)
    assert not (tmp_path / "raw" / "shd" / "shd_train.h5.gz").exists()
