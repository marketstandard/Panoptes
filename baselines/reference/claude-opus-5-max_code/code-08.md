```python
"""A small in-memory cache with per-key time-to-live expiry."""

import threading
import time
from typing import Any, Hashable, Iterator, Optional


class TTLCache:
    """Thread-safe mapping whose entries disappear once their TTL elapses."""

    __slots__ = ("_data", "_lock")

    _MISSING = object()

    def __init__(self) -> None:
        # key -> (expires_at_monotonic, value)
        self._data: dict[Hashable, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def set(self, key: Hashable, value: Any, ttl_seconds: float) -> None:
        """Store `value` under `key`, expiring `ttl_seconds` from now."""
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        # monotonic() is immune to wall-clock adjustments (NTP, DST, manual changes).
        expires_at = time.monotonic() + ttl_seconds
        with self._lock:
            self._data[key] = (expires_at, value)

    def get(self, key: Hashable, default: Any = None) -> Any:
        """Return the live value for `key`, or `default` if missing or expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return default
            expires_at, value = entry
            if expires_at <= now:
                # Drop it on read so an untouched key can't leak forever.
                del self._data[key]
                return default
            return value

    def delete(self, key: Hashable) -> bool:
        """Remove `key`; return True if a live entry was actually removed."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.pop(key, self._MISSING)
            if entry is self._MISSING:
                return False
            expires_at, _ = entry  # type: ignore[misc]
            return expires_at > now

    def purge(self) -> int:
        """Evict every expired entry; return how many were removed."""
        now = time.monotonic()
        with self._lock:
            stale = [k for k, (expires_at, _) in self._data.items() if expires_at <= now]
            for k in stale:
                del self._data[k]
            return len(stale)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __contains__(self, key: object) -> bool:
        return self.get(key, self._MISSING) is not self._MISSING  # type: ignore[arg-type]

    def __len__(self) -> int:
        """Number of live entries. Purges expired ones as a side effect."""
        self.purge()
        with self._lock:
            return len(self._data)

    def __iter__(self) -> Iterator[Hashable]:
        self.purge()
        with self._lock:
            return iter(list(self._data))

    def ttl(self, key: Hashable) -> Optional[float]:
        """Seconds remaining for `key`, or None if it is missing or expired."""
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires_at, _ = entry
            return expires_at - now if expires_at > now else None


# Usage:
#
#   cache = TTLCache()
#   cache.set("session:42", {"user": "dana"}, ttl_seconds=1.5)
#
#   cache.get("session:42")        # -> {'user': 'dana'}
#   "session:42" in cache          # -> True
#   round(cache.ttl("session:42")) # -> 2  (about 1.5s remaining)
#
#   time.sleep(2)
#   cache.get("session:42")        # -> None      (expired: same as never set)
#   cache.get("session:42", "?")   # -> '?'       (your default comes back)
#   "session:42" in cache          # -> False
#   len(cache)                     # -> 0
#   cache.delete("session:42")     # -> False     (nothing live was removed)
```

Design notes:

- **Expiry is lazy, not scheduled.** There is no background thread; an entry is removed the first time it is read after expiring, and `purge()` is available if you want to reclaim memory for keys nobody reads. A pure-lazy cache can grow unboundedly with write-once-never-read keys, so call `purge()` periodically in a long-lived process.
- **"As if never set"** is what drives the details: `get` returns your `default`, `in` is `False`, `ttl` is `None`, `len` doesn't count it, and `delete` reports `False` rather than claiming it removed something real.
- **`time.monotonic()` over `time.time()`** so a clock adjustment can't resurrect an expired entry or expire a fresh one early.
- **The sentinel `_MISSING`** lets `None` be stored as a legitimate cached value while still being distinguishable from a miss.
- **Locking** makes individual operations atomic. It does not make `get`-then-`set` sequences atomic; if you need read-modify-write, add a method that does it under a single lock acquisition.
