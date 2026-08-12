```python
def parse_log_levels(text: str) -> dict[str, int]:
    """Count the lines in `text` that begin with INFO, WARN, or ERROR (case-sensitive)."""
    counts = {"INFO": 0, "WARN": 0, "ERROR": 0}
    for line in text.splitlines():
        for level in counts:
            if line.startswith(level):
                counts[level] += 1
                break
    return counts
```

Notes on the behavior:

- `str.splitlines()` handles `\n`, `\r\n`, and a missing trailing newline without producing a spurious empty final line.
- `startswith` is case-sensitive by default, so `info` or `Error` are ignored as required.
- The `break` keeps a line from being counted twice; with these three prefixes it's not strictly necessary, but it makes the intent explicit if the level list grows.
- All three keys are always present in the result, so callers can index them without a `KeyError` even when a level never appears.

```python
>>> parse_log_levels("INFO started\nWARN disk 91%\nerror ignored\nERROR crash\nINFO done")
{'INFO': 2, 'WARN': 1, 'ERROR': 1}
```
