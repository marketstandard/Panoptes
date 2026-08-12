```python
#!/usr/bin/env python3
"""Print the five largest regular files directly inside a directory."""

import os
import sys

TOP_N = 5
BYTES_PER_MB = 1024 * 1024


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {os.path.basename(argv[0])} DIRECTORY", file=sys.stderr)
        return 2

    path = argv[1]

    if not os.path.exists(path):
        print(f"error: no such directory: {path}", file=sys.stderr)
        return 1
    if not os.path.isdir(path):
        print(f"error: not a directory: {path}", file=sys.stderr)
        return 1

    files = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                try:
                    # follow_symlinks=False keeps a dangling link from raising
                    # and stops us reporting the size of a link's target.
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    files.append((entry.stat(follow_symlinks=False).st_size, entry.name))
                except OSError:
                    continue
    except PermissionError:
        print(f"error: permission denied: {path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"error: could not read {path}: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("no regular files found")
        return 0

    files.sort(key=lambda item: (-item[0], item[1]))

    width = max(len(f"{size / BYTES_PER_MB:.2f}") for size, _ in files[:TOP_N])
    for size, name in files[:TOP_N]:
        print(f"{size / BYTES_PER_MB:>{width}.2f} MB  {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
```

Behavior notes:

- **Not recursive.** `os.scandir` lists one level; subdirectories are skipped rather than descended into.
- **Regular files only.** Directories, symlinks, sockets, and devices are filtered out by `entry.is_file(follow_symlinks=False)`.
- **Missing directory.** A nonexistent path prints a message to stderr and exits `1`; a path that exists but isn't a directory is reported separately, and bad usage exits `2`.
- **Robustness.** Per-entry `OSError` is swallowed so one unreadable file doesn't abort the listing, and ties in size are broken by name so output is deterministic.
- **Megabytes** here means MiB (1024²). Switch `BYTES_PER_MB` to `1_000_000` if you want the decimal definition that disk vendors use.
