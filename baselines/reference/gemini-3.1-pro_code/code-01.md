```python
def parse_log_levels(text: str) -> dict[str, int]:
    """Count the occurrences of lines starting with 'INFO', 'WARN', or 'ERROR' in the given text."""
    counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for line in text.splitlines():
        for level in counts:
            if line.startswith(level):
                counts[level] += 1
                break
    return counts
```