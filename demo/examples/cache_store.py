import time


class CacheStore:
    """
    In-memory key-value cache with a time-to-live for every entry.

    Expired entries are removed lazily, the next time they are read.
    The cache is not thread-safe, so wrap it with a lock when it is shared between threads.

    Args:
        max_size (int): maximum number of entries kept in memory.
        ttl (float): number of seconds before an entry expires.

    Example:
        cache = CacheStore(max_size=100, ttl=30)
        cache.put("user:42", user)
        cache.get("user:42")

    TODO: add an eviction callback so callers can persist evicted entries.
    The current cleanup scans every key, which will be slow for large caches.
    """

    def __init__(self, max_size=1000, ttl=60.0):
        self.max_size = max_size
        self.ttl = ttl
        self._data = {}

    def put(self, key, value):
        """Store value under key."""
        self._data[key] = (value, time.monotonic())

    def get(self, key, default=None):
        """Return the value of key, or default if it is missing or expired."""
        item = self._data.get(key)
        if item is None or time.monotonic() - item[1] > self.ttl:
            self._data.pop(key, None)
            return default
        return item[0]


class TTLPolicy:
    """Decides when a cache entry should expire.

    Attributes:
        default_ttl: number of seconds used when no ttl is given.
    """

    def __init__(self, default_ttl=60.0):
        self.default_ttl = default_ttl
