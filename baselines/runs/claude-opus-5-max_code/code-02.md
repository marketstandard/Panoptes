```python
class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """Fixed-capacity cache that evicts the least recently used entry."""

    def __init__(self, capacity: int):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = capacity
        self._map: dict[object, _Node] = {}

        # Sentinel head/tail remove all null checks from the list surgery.
        # head.next is the most recently used; tail.prev is the least.
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _unlink(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key):
        node = self._map.get(key)
        if node is None:
            return -1
        self._unlink(node)
        self._push_front(node)
        return node.value

    def put(self, key, value) -> None:
        node = self._map.get(key)
        if node is not None:
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return

        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            self._unlink(lru)
            del self._map[lru.key]

        node = _Node(key, value)
        self._map[key] = node
        self._push_front(node)

    def __len__(self) -> int:
        return len(self._map)

    def __contains__(self, key) -> bool:
        return key in self._map
```

**Why this is O(1) on average:** the dictionary gives average-case constant-time lookup from a key to its node, so neither `get` nor `put` ever has to scan the cache to find an entry. Reordering and eviction are done by splicing nodes in a doubly linked list, which touches only a fixed number of pointers regardless of how many items are stored, and the least-recently-used victim is always sitting at `tail.prev` rather than being searched for.

A note on the alternative: `collections.OrderedDict` with `move_to_end` and `popitem(last=False)` gives the same asymptotic behavior in about a dozen lines, and in real code I would reach for it first. The explicit list above is worth writing when you need to see exactly where the recency ordering lives.
