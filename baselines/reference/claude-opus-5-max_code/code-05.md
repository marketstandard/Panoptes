**The bug:** the function returns the second-to-last element of the sorted list rather than the second-largest *distinct* value, so any duplicate of the maximum makes it return the maximum again — `second_largest([5, 5, 3])` gives `5` instead of `3`. For the same reason it returns a value instead of `None` when every element is equal, as in `second_largest([4, 4])`.

**Corrected function:**

```python
def second_largest(values):
    """Return the second-largest distinct value, or None if there isn't one."""
    distinct = sorted(set(values), reverse=True)
    if len(distinct) < 2:
        return None
    return distinct[1]
```

Checks:

```python
second_largest([5, 5, 3])     # 3
second_largest([4, 4])        # None
second_largest([])            # None
second_largest([1, 2, 3, 3])  # 2
second_largest([-1, -2])      # -2
```

Two caveats on `set`: it requires the values to be hashable (fine for numbers) and it will collapse `1` and `1.0`, along with `True`, since they compare equal. If you need to avoid building the set — say the input is a large iterator — a single pass works too:

```python
def second_largest(values):
    """Return the second-largest distinct value, or None if there isn't one."""
    first = second = None
    for v in values:
        if first is None or v > first:
            first, second = v, first
        elif v != first and (second is None or v > second):
            second = v
    return second
```
