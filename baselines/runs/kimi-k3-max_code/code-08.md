```python
import time


class TTLCache:
    """In-memory cache where each entry expires after a per-key TTL."""

    def __init__(self):
        self._store = {}  # key -> (value, expires_at)

    def set(self, key, value, ttl_seconds):
        self._store[key] = (value, time.monotonic() + ttl_seconds)

    def get(self, key):
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._store[key]  # purge so it behaves as never set
            return None
        return value

    def delete(self, key):
        self._store.pop(key, None)


# Usage example:
# cache = TTLCache()
# cache.set("session", {"user": "dana"}, ttl_seconds=60)
# cache.get("session")      # -> {"user": "dana"} within 60 seconds
# cache.get("session")      # -> None after 60 seconds have passed
# cache.set("token", "abc", ttl_seconds=5)
# cache.delete("token")
# cache.get("token")        # -> None
```
