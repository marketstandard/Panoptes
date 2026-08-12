The bug is that sorting keeps duplicates, so when the largest value appears more than once, `values[-2]` equals the largest value itself (e.g., `[1, 3, 3]` returns 3 instead of 1, and `[3, 3]` returns 3 instead of None). Deduplicating before sorting fixes it:

```python
def second_largest(values):
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return None
    return distinct[-2]
```
