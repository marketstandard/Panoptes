The bug occurs because the original function doesn't account for duplicate values; if the largest number appears multiple times in the list, `values[-2]` will incorrectly return the largest number again rather than the true second-largest distinct value. To fix this, we need to remove duplicates before checking the length and finding the second largest.

```python
def second_largest(values):
    unique_values = sorted(list(set(values)))
    if len(unique_values) < 2:
        return None
    return unique_values[-2]
```