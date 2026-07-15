"""Official Zenke Lab downloader; never invoked by tests."""
import gzip, hashlib, shutil, urllib.request
from pathlib import Path
BASE="https://zenkelab.org/datasets/"
FILES={"shd":["shd_train.h5.gz","shd_test.h5.gz"],"ssc":["ssc_train.h5.gz","ssc_valid.h5.gz","ssc_test.h5.gz"]}
def digest(path):
    md5=hashlib.md5();
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024),b""): md5.update(chunk)
    return md5.hexdigest()
def download_dataset(dataset, root="data/raw"):
    if dataset not in FILES: raise ValueError("dataset must be shd or ssc")
    destination=Path(root)/dataset; destination.mkdir(parents=True,exist_ok=True)
    manifest=destination/"md5sums.txt"
    try: urllib.request.urlretrieve(BASE+"md5sums.txt", manifest)
    except OSError as error: raise RuntimeError("could not download official checksum manifest") from error
    sums={line.split()[1].lstrip("*"):line.split()[0] for line in manifest.read_text().splitlines() if len(line.split())>=2}
    for name in FILES[dataset]:
        compressed=destination/name
        if not compressed.exists() or digest(compressed)!=sums.get(name):
            try: urllib.request.urlretrieve(BASE+name, compressed)
            except OSError as error: raise RuntimeError(f"download failed for {name}") from error
        if digest(compressed)!=sums.get(name): raise RuntimeError(f"checksum failed for {name}")
        output=compressed.with_suffix("")
        if not output.exists():
            with gzip.open(compressed,"rb") as source, output.open("wb") as target: shutil.copyfileobj(source,target)
