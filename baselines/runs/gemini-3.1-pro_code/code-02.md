```python
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}

    def get(self, key):
        if key not in self.cache:
            return -1
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value
```

This design leverages Python's built-in dictionary, which maintains insertion order (as of Python 3.7+), to track the most recently used items by moving them to the end upon access or insertion. Because dictionary lookups, insertions, and deletions are on average O(1) time complexity, both `get` and `put` operations meet the strict O(1) requirement.