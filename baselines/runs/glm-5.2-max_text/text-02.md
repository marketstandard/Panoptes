A hash map is a data structure for storing key/value pairs so you can look up a value quickly given its key. The trick is to avoid scanning every entry.

Picture a row of mailboxes in a lobby, each labeled with a number. When you want to deliver a letter to "Alice," you don't walk down the row reading every name — you compute Alice's box number from her name and go straight there. That's exactly what a hash map does: a hash function takes the key and returns an integer, which is mapped onto the array of "buckets" with a modulo. The bucket holds the key/value pair.

Buckets are usually small, so finding the right pair inside one is fast. But sometimes two different keys hash to the same bucket — a collision. There are two common ways to handle this. With separate chaining, each bucket holds a small linked list (or a tiny array) of all the pairs that landed there, and you walk that list to find the right key. With open addressing, if the bucket is taken, you probe the next slot according to a fixed rule (linear probing, quadratic probing, or double hashing) until you find an empty one.

As you insert more pairs, the buckets fill up and collisions get more frequent, which slows lookups. To keep performance good, the hash map tracks its load factor — the ratio of entries to buckets. When the load factor crosses a threshold (often around 0.75), the map resizes: it allocates a larger array (usually double the size) and re-hashes every existing key into its new bucket. That resize is expensive for the one insert that triggers it, but amortized over many inserts it's cheap.

Done well, every operation is O(1) on average.
