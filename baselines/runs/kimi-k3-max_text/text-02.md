A hash map is a data structure that stores key-value pairs so you can retrieve a value almost instantly, given its key. Think of it like a coat check at a busy restaurant: you hand over your coat (the value) and receive a numbered ticket (the key). When you come back, the attendant doesn't search every rack — they walk straight to the spot matching your ticket number.

Under the hood, a hash map keeps an array of slots called buckets. When you insert a key, a hash function converts it into a number, and that number, reduced modulo the array size, selects a bucket. The key-value pair goes into that bucket. Lookups repeat the same calculation and jump directly to the right bucket, which is why they take O(1) time on average.

Sometimes two different keys hash to the same bucket. That's a hash collision, and collisions are unavoidable, so every hash map needs a strategy for them. The most common is chaining: each bucket holds a small list of every pair that landed there, and a lookup scans that short list. An alternative, open addressing, probes nearby empty slots instead.

As entries pile up, buckets fill and collision lists grow longer, which slows everything down. To prevent this, the map tracks its load factor — the ratio of entries to buckets. When the load factor crosses a threshold, often around 0.75, the map resizes: it allocates a larger array, usually double the size, and rehashes every key into the new buckets. Resizing is expensive, but it happens rarely enough that the average cost of insertion stays constant.

In short: hash each key to a bucket, handle collisions gracefully, and grow before things get crowded.
