import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def has_classes(path, classes) -> bool:
    """True if every class folder exists and contains at least one image."""
    path = Path(path)
    if not path.is_dir():
        return False
    for c in classes:
        d = path / c
        if not d.is_dir() or not any(f.suffix.lower() in IMG_EXT for f in d.rglob("*") if f.is_file()):
            return False
    return True


def safe_extract(archive: zipfile.ZipFile, destination) -> None:
    """Reject ZIP entries that would extract outside destination."""
    destination = Path(destination).resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if os.path.commonpath([str(destination), str(target)]) != str(destination):
            raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
    archive.extractall(destination)


def sha256_of(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(raw_dir, classes, file_id: str, expected_sha256: str | None = None) -> Path:
    """Download (Google Drive) and unpack the dataset into raw_dir if it isn't there yet."""
    raw_dir = Path(raw_dir)
    classes = list(classes)
    if has_classes(raw_dir, classes):
        print(f"Dataset already ready: {raw_dir}")
        return raw_dir

    try:
        import gdown
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    work = raw_dir.parent
    zip_path, tmp = work / "data.zip", work / "tmp_unzip"
    shutil.rmtree(raw_dir, ignore_errors=True)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)

    try:
        if not zip_path.is_file():
            if not gdown.download(id=file_id, output=str(zip_path), quiet=False):
                raise RuntimeError("Dataset download failed.")

        digest = sha256_of(zip_path)
        print("data.zip sha256:", digest)
        if expected_sha256 and digest != expected_sha256:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch: expected {expected_sha256}, got {digest}")

        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad is not None:
                zip_path.unlink(missing_ok=True)
                raise RuntimeError(f"Corrupted ZIP member: {bad}. Re-run to download again.")
            safe_extract(archive, tmp)

        source = next((Path(r) for r, dirs, _ in os.walk(tmp) if set(classes).issubset(dirs)), None)
        if source is None or not has_classes(source, classes):
            raise FileNotFoundError("Could not find all class folders with images in the ZIP.")

        raw_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(raw_dir))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"Dataset ready: {raw_dir}")
    return raw_dir