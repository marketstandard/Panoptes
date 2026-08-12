```python
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()

    if not args.directory.is_dir():
        print(f"Directory does not exist: {args.directory}", file=sys.stderr)
        return 1

    files = [
        (entry.stat().st_size, entry)
        for entry in args.directory.iterdir()
        if entry.is_file()
    ]

    for size, path in sorted(files, reverse=True)[:5]:
        print(f"{path.name}	{size / (1024 * 1024):.2f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```