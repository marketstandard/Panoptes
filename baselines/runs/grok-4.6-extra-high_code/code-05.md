The bug is that sorting keeps duplicate values, so `values[-2]` is not the second-largest *distinct* number (for example `[3, 3, 1]` returns `3` instead of `1`). If every value is the same, it also returns that value instead of `None`.

```python
def second_largest(values):
    unique = sorted(set(values))
    if len(unique) < 2:
        return None
    return unique[-2]
```
