"""Create a 2,000-patient sample of PhysioNet Challenge 2019 training_setA.

The source URL is a single archive containing every training_setA patient, so
the default command first checks its size. It downloads and extracts the first
300 PSV files only when the archive is at or below the configured size limit.
Otherwise it prints the manual, bandwidth-safe workflow instead.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ARCHIVE_URL = "https://archive.physionet.org/challenge/2019/training_setA.zip"
SAMPLE_SIZE = 2_000
DEFAULT_MAX_ARCHIVE_MB = 200
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def remote_size_bytes(url: str) -> int | None:
    """Return the advertised archive size without downloading its body."""
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            value = response.headers.get("Content-Length")
    except OSError as error:
        print(f"Could not check archive size: {error}", file=sys.stderr)
        return None
    return int(value) if value and value.isdigit() else None


def extract_sample(archive: Path, destination: Path, count: int, force: bool) -> None:
    """Copy the first `count` PSV members of an archive into `destination`."""
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        members = sorted(
            member
            for member in zip_file.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".psv")
        )[:count]

        if len(members) < count:
            raise ValueError(f"Archive contains only {len(members)} PSV files; expected {count}.")

        for member in members:
            target = destination / Path(member.filename).name
            if target.exists() and not force:
                raise FileExistsError(
                    f"{target} already exists. Use --force to replace sample files."
                )
            with zip_file.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)

    print(f"Saved {len(members)} patient files to {destination}")


def print_manual_instructions(size_bytes: int | None) -> None:
    size_note = (
        f"The training_setA archive is approximately {size_bytes / 1024**3:.1f} GB."
        if size_bytes
        else "The archive size could not be determined."
    )
    print(
        f"{size_note}\n\n"
        "To keep this prototype fast, the full archive will not be downloaded.\n"
        f"1. Download {ARCHIVE_URL} in a browser.\n"
        "2. Extract it locally.\n"
        "3. Either copy the first 2,000 .psv files into data/raw/, or run:\n"
        "   python src/00_download_sample.py --archive PATH\\TO\\training_setA.zip\n"
        "   (The --archive option extracts only the first 2,000 patient files.)"
    )


def download_archive(url: str, destination: Path) -> None:
    print(f"Downloading {url} ...")
    with urllib.request.urlopen(url, timeout=60) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="Existing training_setA.zip to sample.")
    parser.add_argument("--force", action="store_true", help="Replace existing sample files.")
    parser.add_argument(
        "--max-archive-mb",
        type=int,
        default=DEFAULT_MAX_ARCHIVE_MB,
        help="Download the source only when it is no larger than this many MB (default: 200).",
    )
    args = parser.parse_args()

    if args.archive:
        extract_sample(args.archive, RAW_DIR, SAMPLE_SIZE, args.force)
        return

    size_bytes = remote_size_bytes(ARCHIVE_URL)
    max_bytes = args.max_archive_mb * 1024**2
    if size_bytes is None or size_bytes > max_bytes:
        print_manual_instructions(size_bytes)
        return

    with tempfile.TemporaryDirectory() as directory:
        archive = Path(directory) / "training_setA.zip"
        download_archive(ARCHIVE_URL, archive)
        extract_sample(archive, RAW_DIR, SAMPLE_SIZE, args.force)


if __name__ == "__main__":
    main()
