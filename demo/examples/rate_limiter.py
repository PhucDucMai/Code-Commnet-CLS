import time


class RateLimiter:
    """
    Limit how often a function can be called within a time window.

    Args:
        max_calls (int): maximum number of calls allowed per window.
        period (float): length of the window in seconds.

    Example:
        limiter = RateLimiter(max_calls=5, period=1.0)
        if limiter.allow():
            send_request()

    TODO: the window is fixed, not sliding. A sliding window would be
    more accurate but needs more memory.
    """

    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self._calls = []

    def allow(self):
        """Return True if a call is allowed now."""
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < self.period]
        if len(self._calls) < self.max_calls:
            self._calls.append(now)
            return True
        return False


class NoLimit(RateLimiter):
    """A limiter that always allows calls. Useful in unit tests."""

    def __init__(self):
        super().__init__(max_calls=0, period=0)

    def allow(self):
        return True
