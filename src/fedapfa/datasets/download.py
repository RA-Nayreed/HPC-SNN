"""Atomic, checksummed downloader for official Zenke Lab SHD and SSC files."""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import os
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

from .validation import OFFICIAL_EXPECTATIONS, DatasetValidationError, validate_hdf5

BASE_URL = "https://zenkelab.org/datasets"
DATASET_FILES = {
    "shd": ("shd_train.h5.gz", "shd_test.h5.gz"),
    "ssc": ("ssc_train.h5.gz", "ssc_valid.h5.gz", "ssc_test.h5.gz"),
}


class DownloadError(RuntimeError):
    pass


def md5_digest(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksums(text: str) -> dict[str, str]:
    checksums = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 32:
            checksums[parts[-1].lstrip("*")] = parts[0].lower()
    return checksums


def _url_download(url: str, destination: Path) -> None:
    try:
        with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
            shutil.copyfileobj(response, output)
    except (OSError, urllib.error.URLError) as error:
        raise DownloadError(f"download failed for {url}: {error}") from error


@contextlib.contextmanager
def _file_lock(path: Path, timeout: float = 120.0) -> Iterator[None]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.write(descriptor, str(os.getpid()).encode())
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise DownloadError(f"timed out waiting for download lock {path}") from None
            time.sleep(0.1)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def _atomic_download(url: str, destination: Path, downloader: Callable[[str, Path], None]) -> None:
    part = destination.with_name(destination.name + ".part")
    part.unlink(missing_ok=True)
    try:
        downloader(url, part)
        if not part.is_file():
            raise DownloadError(f"downloader did not create {part}")
        os.replace(part, destination)
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError(f"download failed for {url}: {error}") from error
    finally:
        part.unlink(missing_ok=True)


def _atomic_decompress(compressed: Path, output: Path) -> None:
    part = output.with_name(output.name + ".part")
    part.unlink(missing_ok=True)
    try:
        with gzip.open(compressed, "rb") as source, part.open("wb") as target:
            shutil.copyfileobj(source, target)
        validate_hdf5(part, OFFICIAL_EXPECTATIONS[output.name])
        os.replace(part, output)
    except (OSError, DatasetValidationError) as error:
        raise DownloadError(f"failed to extract valid HDF5 file from {compressed}: {error}") from error
    finally:
        part.unlink(missing_ok=True)


def download_dataset(
    dataset: str, root: str | Path = "data/raw", downloader: Callable[[str, Path], None] = _url_download
) -> list[Path]:
    if dataset not in DATASET_FILES:
        raise ValueError("dataset must be shd or ssc")
    destination = Path(root) / dataset
    destination.mkdir(parents=True, exist_ok=True)
    with _file_lock(destination / ".download.lock"):
        manifest = destination / "md5sums.txt"
        _atomic_download(f"{BASE_URL}/md5sums.txt", manifest, downloader)
        checksums = parse_checksums(manifest.read_text(encoding="utf-8"))
        outputs = []
        for filename in DATASET_FILES[dataset]:
            expected = checksums.get(filename)
            if expected is None:
                raise DownloadError(f"official checksum manifest has no entry for {filename}")
            compressed = destination / filename
            if not compressed.is_file() or md5_digest(compressed) != expected:
                _atomic_download(f"{BASE_URL}/{filename}", compressed, downloader)
            actual = md5_digest(compressed)
            if actual != expected:
                compressed.unlink(missing_ok=True)
                raise DownloadError(f"MD5 mismatch for {filename}: expected {expected}, got {actual}")
            output = compressed.with_suffix("")
            try:
                validate_hdf5(output, OFFICIAL_EXPECTATIONS[output.name])
            except (FileNotFoundError, DatasetValidationError):
                _atomic_decompress(compressed, output)
            outputs.append(output)
        return outputs
