```python
def parse_log_levels(text: str) -> dict[str, int]:
    """Return a count of lines beginning with INFO, WARN, or ERROR."""
    counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for line in text.splitlines():
        for level in counts:
            if line.startswith(level):
                counts[level] += 1
                break
    return counts
```
