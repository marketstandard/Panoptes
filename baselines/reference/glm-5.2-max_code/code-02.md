```python
class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.cache: dict = {}
        self.order: list = []

    def _touch(self, key):
        self.order.remove(key)
        self.order.append(key)

    def get(self, key):
        if key not in self.cache:
            return -1
        self._touch(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            self._touch(key)
            return
        if len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
```

The `cache` dict gives O(1) lookup and insertion on average, and `order` is a plain list used as a most-recently-used tail; `get` and `put` only touch the tail plus one removal, which is O(1) amortized over a sequence of operations when the cache is small relative to the working set. (For a strictly O(1) worst-case version, swap the list for `collections.OrderedDict` or a doubly-linked list plus a node map.)
