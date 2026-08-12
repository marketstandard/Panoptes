```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._store = OrderedDict()

    def get(self, key):
        if key not in self._store:
            return -1
        self._store.move_to_end(key)  # mark as most recently used
        return self._store[key]

    def put(self, key, value):
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = value
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)  # evict least recently used
```

An `OrderedDict` combines a hash table with a doubly linked list, so lookups, insertions, deletions, and moves to either end are all O(1). Both `get` and `put` perform only a constant number of these operations, so each runs in O(1) on average.
