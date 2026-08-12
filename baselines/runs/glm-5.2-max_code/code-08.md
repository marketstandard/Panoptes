```python
import time
from typing import Any, Optional


class TTLCache:
    """A simple in-memory cache with per-entry time-to-live, stdlib only."""

    def __init__(self) -> None:
        self._store: dict[Any, tuple[Any, float]] = {}

    def set(self, key: Any, value: Any, ttl_seconds: float) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must be non-negative")
        expires_at = time.monotonic() + ttl_seconds
        self._store[key] = (value, expires_at)

    def get(self, key: Any) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._store[key]
            return None
        return value

    def delete(self, key: Any) -> None:
        self._store.pop(key, None)

    def cleanup(self) -> int:
        """Remove all expired entries. Returns the number removed."""
        now = time.monotonic()
        expired = [k for k, (_, exp) in self._store.items() if now >= exp]
        for k in expired:
            del self._store[k]
        return len(expired)


# Usage example:
# cache = TTLCache()
# cache.set("api_key", "abc123", ttl_seconds=60)
# print(cache.get("api_key"))  # -> "abc123" within 60s, None after
# cache.delete("api_key")
```
