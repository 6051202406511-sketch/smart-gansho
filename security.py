"""Sanitization and best-effort cleanup for app-managed private copies.

No portable overwrite routine can guarantee erasure on SSDs, copy-on-write
filesystems, backups, snapshots, journals, or storage controlled by a device.
Callers must only pass files that the operator has agreed to delete.
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import unicodedata
import uuid
from pathlib import Path

ALLOWED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


def sanitize_text(text: object, *, max_length: int = 20_000) -> str:
    value = str(text) if text is not None else ""
    cleaned = "".join(
        character
        for character in value
        if character in "\t\n\r" or unicodedata.category(character) != "Cc"
    )
    return cleaned.strip()[:max_length]


def safe_excel_value(value: object) -> str:
    text = sanitize_text(value)
    if text.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + text
    return text


def create_private_cache(parent: str | Path | None = None) -> Path:
    temp_root = Path(tempfile.gettempdir()).resolve()
    base = Path(parent).resolve() if parent else temp_root
    try:
        base.relative_to(temp_root)
    except ValueError as error:
        raise ValueError("Private cache must be created beneath the system temp directory.") from error
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="school-application-", dir=base))
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def stage_private_copy(
    source: str | Path,
    cache_dir: str | Path,
    *,
    max_bytes: int = 40 * 1024 * 1024,
) -> Path:
    original = Path(source)
    cache = Path(cache_dir).resolve()
    if original.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
        raise ValueError("PNG/JPG/JPEG/PDF ファイルを選択してください。")
    if not original.is_file():
        raise FileNotFoundError(f"ファイルが見つかりません: {original}")
    if original.stat().st_size > max_bytes:
        raise ValueError("ファイルが大きすぎます（上限40MB）。")
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / f"{uuid.uuid4().hex}{original.suffix.lower()}"
    shutil.copyfile(original, destination)
    try:
        destination.chmod(0o600)
    except OSError:
        pass
    return destination


def _overwrite_file(path: Path, passes: int) -> None:
    size = path.stat().st_size
    with path.open("r+b", buffering=0) as file:
        for pass_index in range(passes):
            file.seek(0)
            remaining = size
            while remaining:
                block_size = min(1024 * 1024, remaining)
                data = (
                    b"\x00" * block_size
                    if pass_index % 2 == 0
                    else secrets.token_bytes(block_size)
                )
                file.write(data)
                remaining -= block_size
            file.flush()
            os.fsync(file.fileno())


def shred_file(file_path: str | Path, *, passes: int = 3) -> tuple[bool, str]:
    """Best-effort zero/random overwrite and unlink of one regular file."""
    target = Path(file_path)
    if passes < 1:
        raise ValueError("passes must be at least one")
    try:
        if target.is_symlink() or not target.is_file():
            return False, f"Not a regular file: {target}"
        if target.stat().st_nlink > 1:
            return False, f"Refusing to overwrite a file with multiple hard links: {target}"
        _overwrite_file(target, passes)
        target.unlink()
    except OSError as error:
        return False, f"Could not overwrite and remove {target}: {error}"
    return (
        True,
        "Temporary copy overwritten and removed on a best-effort basis; "
        "physical erasure cannot be guaranteed on all storage/filesystems.",
    )


def cleanup_cache_folder(cache_dir: str | Path) -> list[str]:
    """Shred files beneath the exact app cache and remove its directories."""
    candidate = Path(cache_dir).absolute()
    temp_root = Path(tempfile.gettempdir()).resolve()
    if candidate.is_symlink():
        return ["Refusing cleanup outside an app-owned temporary cache directory."]
    root = candidate.resolve()
    if not root.name.startswith("school-application-"):
        return ["Refusing cleanup outside an app-owned temporary cache directory."]
    try:
        root.relative_to(temp_root)
    except ValueError:
        return ["Refusing cleanup outside an app-owned temporary cache directory."]
    if not root.exists():
        return []
    errors: list[str] = []
    if root.is_symlink() or not root.is_dir():
        return [f"Cache path is not a regular directory: {root}"]

    for current, dirs, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        safe_dirs: list[str] = []
        for directory in dirs:
            candidate = current_path / directory
            if candidate.is_symlink():
                try:
                    candidate.unlink()
                except OSError as error:
                    errors.append(f"{candidate}: {error}")
            else:
                safe_dirs.append(directory)
        dirs[:] = safe_dirs
        for filename in files:
            target = current_path / filename
            if target.is_symlink():
                try:
                    target.unlink()
                except OSError as error:
                    errors.append(f"{target}: {error}")
                continue
            ok, message = shred_file(target)
            if not ok:
                errors.append(message)

    for current, dirs, _files in os.walk(root, topdown=False, followlinks=False):
        for directory in dirs:
            target = Path(current) / directory
            try:
                target.rmdir()
            except OSError as error:
                errors.append(f"{target}: {error}")
    try:
        root.rmdir()
    except OSError as error:
        errors.append(f"{root}: {error}")
    return errors


secure_delete = shred_file
secure_delete_best_effort = shred_file
