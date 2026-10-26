import time


class ExponentialBackoff:
    """
    Retry a callable with exponentially growing delays between attempts.

    Each failed attempt doubles the delay, up to a maximum.

    Parameters
    ----------
    base_delay : float
        Delay in seconds before the first retry.
    max_delay : float
        Upper bound for the delay in seconds.
    max_attempts : int
        Number of attempts before the last error is raised.

    Examples
    --------
    backoff = ExponentialBackoff(base_delay=0.5, max_attempts=3)
    result = backoff.run(fetch_page, url)

    Notes
    -----
    The delays are not randomized yet, so many clients retrying at the same time will hit the server together.
    """

    def __init__(self, base_delay=1.0, max_delay=30.0, max_attempts=5):
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_attempts = max_attempts

    def run(self, func, *args):
        delay = self.base_delay
        for attempt in range(self.max_attempts):
            try:
                return func(*args)
            except Exception:
                if attempt == self.max_attempts - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)
