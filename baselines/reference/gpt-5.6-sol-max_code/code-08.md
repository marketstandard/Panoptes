```python
import time
from typing import Any


class TTLCache:
    def __init__(self) -> None:
        self._entries: dict[Any, tuple[Any, float]] = {}

    def set(self, key: Any, value: Any, ttl_seconds: float) -> None:
        self._entries[key] = (value, time.monotonic() + ttl_seconds)

    def get(self, key: Any) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None

        value, expires_at = entry
        if time.monotonic() >= expires_at:
            del self._entries[key]
            return None
        return value

    def delete(self, key: Any) -> None:
        self._entries.pop(key, None)


# cache = TTLCache()
# cache.set("token", "abc123", ttl_seconds=30)
# print(cache.get("token"))  # "abc123" until it expires, then None
# cache.delete("token")
```