The function sorts all values without removing duplicates, so its penultimate item may equal the maximum rather than being the second-largest distinct value.

```python
def second_largest(values):
    distinct = sorted(set(values))
    if len(distinct) < 2:
        return None
    return distinct[-2]
```