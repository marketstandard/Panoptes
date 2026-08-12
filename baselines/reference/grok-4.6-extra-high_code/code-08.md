```python
import time


class TTLCache:
    def __init__(self):
        self._store = {}

    def _expired(self, expires_at: float) -> bool:
        return time.monotonic() >= expires_at

    def set(self, key, value, ttl_seconds):
        if ttl_seconds <= 0:
            self._store.pop(key, None)
            return
        self._store[key] = (value, time.monotonic() + ttl_seconds)

    def get(self, key):
        item = self._store.get(key)
        if item is None:
            return None
        value, expires_at = item
        if self._expired(expires_at):
            del self._store[key]
            return None
        return value

    def delete(self, key):
        self._store.pop(key, None)


# cache = TTLCache()
# cache.set("session", {"user": "ada"}, ttl_seconds=30)
# cache.get("session")  # dict if still within 30s, else None
# cache.delete("session")
```
