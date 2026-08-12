```python
#!/usr/bin/env python3
import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the five largest regular files directly inside a directory."
    )
    parser.add_argument("directory", help="Path to the directory to inspect")
    args = parser.parse_args()

    directory = args.directory

    if not os.path.isdir(directory):
        print(f"error: '{directory}' is not a directory or does not exist", file=sys.stderr)
        return 1

    files = []
    try:
        for name in os.listdir(directory):
            full = os.path.join(directory, name)
            if os.path.isfile(full) and not os.path.islink(full):
                size = os.path.getsize(full)
                files.append((name, size))
    except OSError as e:
        print(f"error: could not list '{directory}': {e}", file=sys.stderr)
        return 1

    files.sort(key=lambda item: item[1], reverse=True)

    for name, size in files[:5]:
        mb = size / (1024 * 1024)
        print(f"{mb:.2f} MB\t{name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
