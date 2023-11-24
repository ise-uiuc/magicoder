from pathlib import Path

import wget as _wget


def wget(url: str, path: Path | None = None) -> Path:
    if path is None:
        filename = _wget.detect_filename(url)
        path = Path(filename)
    if not path.exists():
        _wget.download(url, path.as_posix())
    return path
