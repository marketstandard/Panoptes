A hash map (also called a hash table, or a dictionary in Python) stores key–value pairs and lets you retrieve a value by its key in roughly constant time, no matter how much data you've put in it.

Think of a coat check at a theater. Instead of searching every hanger for your jacket, you hand over a ticket, and the number on it sends the attendant straight to one specific rack. A hash map works the same way, except the "ticket number" is computed from the key itself.

**Buckets and the hash function**

Internally, a hash map is backed by an array of slots called buckets. When you insert a key, a hash function turns that key into an integer, and the map takes that integer modulo the array's size to choose a bucket index. Because the same key always hashes to the same number, a lookup just repeats the identical computation and jumps directly to the right bucket — one step, not a scan through everything.

**Collisions**

Two different keys can land in the same bucket. That's a collision, and it is unavoidable: there are vastly more possible keys than there are buckets. The usual remedy is chaining, where each bucket holds a short list of entries and the map walks that list comparing keys for equality. Note the division of labor — the hash finds the bucket, and an equality check confirms the actual key. An alternative strategy is open addressing, where a colliding entry probes forward to the next free slot.

**Resizing**

As entries accumulate, chains grow longer and lookups start drifting toward linear time. So a hash map watches its load factor — the number of entries divided by the number of buckets — and when that crosses a threshold (often around 0.75), it allocates a bigger array, typically double the size, and rehashes every existing entry into the new one. Resizing is expensive, but it happens rarely enough that the average cost per insertion stays constant.
