```python
#!/usr/bin/env python3
import os
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: largest_files.py DIRECTORY", file=sys.stderr)
        return 2
    path = sys.argv[1]
    if not os.path.isdir(path):
        print(f"Directory does not exist: {path}", file=sys.stderr)
        return 1
    files = []
    with os.scandir(path) as entries:
        for entry in entries:
            if not entry.is_file(follow_symlinks=False):
                continue
            try:
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
            files.append((size, entry.name))
    files.sort(key=lambda item: item[0], reverse=True)
    for size, name in files[:5]:
        mb = size / (1024 * 1024)
        print(f"{name}\t{mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
