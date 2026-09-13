"""Download only the three raw MaleCNS v1.0 tables, atomically and with checksums."""

import argparse
import base64
import hashlib
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
DEFAULT_DATA_DIR = Path("data/malecns-v1.0")
FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
CHUNK_SIZE = 1024 * 1024


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return base64.b64encode(hashlib.file_digest(stream, "md5").digest()).decode()


def download_file(session: requests.Session, filename: str, directory: Path) -> Path:
    """Verify against GCS Content-Length and MD5 before promoting a .part file.

    Existing valid files are reused. Interrupted/corrupt downloads are retried by
    rerunning the command; partial files are never accepted as complete tables.
    """
    if filename not in FILES.values():
        raise ValueError(f"Not a v0.1 dataset file: {filename}")
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    url = BASE_URL + filename
    with session.head(url, timeout=(15, 60), allow_redirects=True) as response:
        response.raise_for_status()
        size = int(response.headers["Content-Length"])
        hashes = dict(
            item.strip().split("=", 1)
            for item in response.headers.get("x-goog-hash", "").split(",")
            if "=" in item
        )
        expected_md5 = hashes.get("md5")
        generation = response.headers.get("x-goog-generation")
    if not expected_md5 or size <= 0:
        raise ValueError(f"Missing upstream size/MD5 metadata for {filename}")
    if destination.is_file() and destination.stat().st_size == size:
        if _digest(destination) == expected_md5:
            print(f"Verified existing: {filename}")
            return destination

    # Pin the GET to the generation whose checksum we inspected.
    params = {"generation": generation} if generation else None
    partial = destination.with_suffix(destination.suffix + ".part")
    checksum = hashlib.md5()
    downloaded = 0
    try:
        with session.get(url, params=params, stream=True, timeout=(15, 120)) as response:
            response.raise_for_status()
            with (
                partial.open("wb") as stream,
                tqdm(
                    total=size, unit="B", unit_scale=True, desc=filename, mininterval=2
                ) as progress,
            ):
                for chunk in response.iter_content(CHUNK_SIZE):
                    stream.write(chunk)
                    checksum.update(chunk)
                    downloaded += len(chunk)
                    progress.update(len(chunk))
        actual_md5 = base64.b64encode(checksum.digest()).decode()
        if downloaded != size or actual_md5 != expected_md5:
            raise ValueError(f"Size/MD5 verification failed: {filename}")
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Downloaded and verified: {filename}")
    return destination


def download_dataset(directory: Path = DEFAULT_DATA_DIR) -> list[Path]:
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=retry))
        return [download_file(session, filename, directory) for filename in FILES.values()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    try:
        download_dataset(args.data_dir)
    except (OSError, ValueError, requests.RequestException) as error:
        parser.exit(1, f"Download failed: {error}\n")


if __name__ == "__main__":
    main()
