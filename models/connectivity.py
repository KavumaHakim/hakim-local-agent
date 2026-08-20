"""Is there internet?

Hosted models need it and local ones do not, so this decides whether the cloud
options are offered or greyed out. It is asked often - every model listing -
so it must be cheap and must never be the reason a request feels slow.

Deliberately a raw TCP connect to a well-known address rather than an HTTP
request to a provider:

* No DNS lookup, which is itself a slow failure when the network is down.
* No API key, so it works before one is configured and costs no quota.
* Provider *reachability* is a different question from *internet*, and
  conflating them turns a provider outage into "you are offline", which sends
  someone looking at their router instead of at a status page.

The answer is cached, because a connectivity check per model per listing would
add up to exactly the sort of latency that made /api/models an 18-second
request the last time something was probed too eagerly.
"""

from __future__ import annotations

import socket
import threading
import time

# Addresses rather than hostnames, so no DNS is required. Two, so one blocked
# resolver or one blackholed route does not read as "offline".
PROBES: tuple[tuple[str, int], ...] = (
    ("1.1.1.1", 443),
    ("8.8.8.8", 443),
)

# Long enough to keep listings cheap, short enough that plugging the network
# back in is noticed without a restart.
TTL_SECONDS = 30.0

# A connect that has not completed in this long is not a usable connection for
# a hosted model anyway.
TIMEOUT_SECONDS = 1.5


class Connectivity:
    """Cached answer to 'is the internet up'."""

    def __init__(
        self,
        probes: tuple[tuple[str, int], ...] = PROBES,
        ttl: float = TTL_SECONDS,
        timeout: float = TIMEOUT_SECONDS,
    ) -> None:
        self._probes = probes
        self._ttl = ttl
        self._timeout = timeout
        self._lock = threading.Lock()
        self._checked_at = 0.0
        self._online = False

    def online(self, *, force: bool = False) -> bool:
        """Whether a hosted model could be reached right now."""
        with self._lock:
            fresh = time.monotonic() - self._checked_at < self._ttl
            if fresh and not force:
                return self._online
            self._online = self._probe()
            self._checked_at = time.monotonic()
            return self._online

    def invalidate(self) -> None:
        """Force the next call to re-check.

        Called after a hosted request fails, so the UI stops offering cloud
        models the moment one actually fails rather than up to a TTL later.
        """
        with self._lock:
            self._checked_at = 0.0

    def _probe(self) -> bool:
        for host, port in self._probes:
            try:
                with socket.create_connection((host, port), timeout=self._timeout):
                    return True
            except OSError:
                continue
        return False
