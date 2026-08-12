```python
from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity: int):
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key):
        if key not in self._data:
            return -1
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)
```

`OrderedDict` is a hash map plus a doubly linked list, so lookup, move-to-end, insert, and eviction of the oldest item are all average O(1). Marking a key as most-recently-used is a pointer update rather than a scan, which is why both `get` and `put` meet the complexity requirement.
