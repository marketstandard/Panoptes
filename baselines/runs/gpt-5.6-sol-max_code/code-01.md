```python
def parse_log_levels(text: str) -> dict[str, int]:
    """Count lines that begin with a recognized log level."""
    counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for line in text.splitlines():
        for level in counts:
            if line.startswith(level):
                counts[level] += 1
                break
    return counts
```