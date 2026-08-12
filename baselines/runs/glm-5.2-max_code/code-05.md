The bug is that `sorted(values)[-2]` returns the second element from the end of the sorted list, which is the second-largest only when all values are distinct; if the largest value appears more than once (for example `[5, 5, 3]`), the function returns `5` instead of `3`, the true second-largest distinct value. The fix is to deduplicate first.

```python
def second_largest(values):
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return None
    return distinct[-2]
```
