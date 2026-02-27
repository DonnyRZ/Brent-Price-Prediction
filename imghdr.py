"""Compatibility shim for Python 3.13 where stdlib imghdr was removed.

Streamlit 1.19 still imports `imghdr`. Provide a minimal `what()` implementation
based on Pillow so the app can boot on newer runtimes.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path


def _read_bytes(file) -> bytes | None:
    if file is None:
        return None
    if hasattr(file, "read"):
        try:
            pos = file.tell()
        except Exception:
            pos = None
        data = file.read()
        if pos is not None:
            try:
                file.seek(pos)
            except Exception:
                pass
        return data
    try:
        return Path(file).read_bytes()
    except Exception:
        return None


def what(file, h=None):
    data = h if h is not None else _read_bytes(file)
    if not data:
        return None
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(data)) as img:
            fmt = img.format
        return fmt.lower() if fmt else None
    except Exception:
        return None


# Keep an empty tests list for compatibility with stdlib imghdr API.
tests = []
