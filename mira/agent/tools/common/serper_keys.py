import itertools
import threading
import time

__all__ = ["SerperKeyPool"]


class SerperKeyPool:
    def __init__(self, keys, cooldown_s: float = 60.0):
        self._keys = [k.strip() for k in (keys or []) if k and k.strip()]
        self._cycle = itertools.cycle(self._keys) if self._keys else None
        self._cooling = {}
        self._lock = threading.Lock()
        self.cooldown_s = float(cooldown_s)

    def __bool__(self) -> bool:
        return bool(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def acquire(self) -> str:
        if not self._keys:
            return ""
        with self._lock:
            now = time.monotonic()
            for _ in range(len(self._keys)):
                key = next(self._cycle)
                if self._cooling.get(key, 0.0) <= now:
                    return key
            return min(self._keys, key=lambda k: self._cooling.get(k, 0.0))

    def rate_limited(self, key: str, retry_after: float = 0.0) -> None:
        if not key:
            return
        cooldown = retry_after if retry_after and retry_after > 0 else self.cooldown_s
        with self._lock:
            self._cooling[key] = time.monotonic() + min(cooldown, 300.0)

    def any_available(self) -> bool:
        with self._lock:
            now = time.monotonic()
            return any(self._cooling.get(k, 0.0) <= now for k in self._keys)
