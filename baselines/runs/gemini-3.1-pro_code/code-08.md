```python
import time

class TTLCache:
    def __init__(self):
        self.cache = {}

    def set(self, key, value, ttl_seconds):
        expires_at = time.time() + ttl_seconds
        self.cache[key] = (value, expires_at)

    def get(self, key):
        if key in self.cache:
            value, expires_at = self.cache[key]
            if time.time() < expires_at:
                return value
            else:
                # Clean up expired key on access
                del self.cache[key]
        return None

    def delete(self, key):
        if key in self.cache:
            # Check expiration so deleting an already expired key does nothing
            _, expires_at = self.cache[key]
            del self.cache[key]
            if time.time() >= expires_at:
                raise KeyError(key)

# Usage Example:
# cache = TTLCache()
# cache.set('username', 'alice', 2) # expires in 2 seconds
# print(cache.get('username'))      # Output: 'alice'
# time.sleep(3)
# print(cache.get('username'))      # Output: None
```