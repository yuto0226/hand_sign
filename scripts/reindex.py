"""重新將目錄內 sX_XXXX.jpg 格式的照片從 0000 開始連續編號。

Usage:
    python reindex.py data/cnn/tora
"""

from __future__ import annotations

import argparse
import re
import uuid
from pathlib import Path

_PATTERN = re.compile(r"^(.+?)_(\d+)(\.[^.]+)$")


def reindex(directory: Path) -> None:
    files: list[tuple[int, Path]] = []
    for f in directory.iterdir():
        m = _PATTERN.match(f.name)
        if m:
            files.append((int(m.group(2)), f))

    if not files:
        print(f"[!] 找不到符合格式的檔案：{directory}")
        return

    files.sort(key=lambda x: x[0])

    first_match = _PATTERN.match(files[0][1].name)
    assert first_match is not None
    prefix = first_match.group(1)
    suffix = first_match.group(3)

    tmp_paths: list[tuple[Path, Path]] = []
    for _, src in files:
        tmp = src.with_name(f"_tmp_{uuid.uuid4().hex}{suffix}")
        tmp_paths.append((src, tmp))
        src.rename(tmp)

    for i, (_, tmp) in enumerate(tmp_paths):
        dst = directory / f"{prefix}_{i:04d}{suffix}"
        print(f"  {tmp.name} → {dst.name}")
        tmp.rename(dst)

    print(f"[*] 完成，共重新編號 {len(files)} 個檔案 → {directory}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"[!] 不是目錄：{args.directory}")
        return

    reindex(args.directory)


if __name__ == "__main__":
    main()
