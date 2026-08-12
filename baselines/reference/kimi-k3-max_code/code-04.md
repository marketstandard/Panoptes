```python
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: largest_files.py <directory>")
    directory = Path(sys.argv[1])
    if not directory.is_dir():
        sys.exit(f"error: {directory} does not exist or is not a directory")
    files = (p for p in directory.iterdir() if p.is_file())
    largest = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:5]
    for path in largest:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{size_mb:8.2f} MB  {path.name}")


if __name__ == "__main__":
    main()
```
