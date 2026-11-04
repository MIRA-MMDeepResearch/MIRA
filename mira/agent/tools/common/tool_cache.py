import atexit
import hashlib
import json
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
import time
from collections import OrderedDict
from queue import Empty, Full, Queue
from typing import Optional

from .web_errors import is_web_failure, sanitize_url

__all__ = [
    "configure",
    "enabled",
    "key_for",
    "is_cacheable_text",
    "get_json",
    "put_json",
    "get_blob",
    "put_blob",
    "stats",
    "NS_BROWSER_SEARCH",
    "NS_BROWSER_FETCH",
    "NS_IMAGE_LENS",
    "NS_IMAGE_BLOB",
]

NS_BROWSER_SEARCH = "browser_search"
NS_BROWSER_FETCH = "browser_fetch"
NS_IMAGE_LENS = "image_lens"
NS_IMAGE_BLOB = "image_blob"

_ALL_NAMESPACES = (
    NS_BROWSER_SEARCH,
    NS_BROWSER_FETCH,
    NS_IMAGE_LENS,
    NS_IMAGE_BLOB,
)

@dataclass(frozen=True)
class _Settings:
    enabled: bool
    read: bool
    write: bool
    run_dir: Optional[str]
    global_dir: Optional[str]
    shards: int
    blob_max_bytes: int
    mem_max_bytes: int
    mem_stripe_max: int
    queue_max_bytes: int


_SETTINGS = _Settings(
    enabled=False, read=True, write=True, run_dir=None, global_dir=None, shards=32,
    blob_max_bytes=4 * 1024 * 1024,
    mem_max_bytes=256 * 1024 * 1024,
    mem_stripe_max=(256 * 1024 * 1024) // 16,
    queue_max_bytes=256 * 1024 * 1024,
)

_KEY_VERSIONS = {
    NS_BROWSER_SEARCH: 1,
    NS_BROWSER_FETCH: 1,
    NS_IMAGE_LENS: 1,
    NS_IMAGE_BLOB: 1,
}

_STORES = {}
_STORES_LOCK = threading.Lock()
_HITS = {}
_MISSES = {}
_STAT_LOCK = threading.Lock()


def configure(cfg) -> None:
    global _SETTINGS, _QUEUED_BYTES, _DROPPED
    flush(timeout=None)
    _mem_clear()
    mem_max = max(0, int(cfg.cache.tool_mem_max_bytes))
    run_dir = cfg.cache.dir
    global_dir = cfg.cache.global_dir
    enabled_flag = bool(cfg.cache.tool_enabled)
    run_dir = str(run_dir) if run_dir else None
    if enabled_flag and not run_dir:
        print("[tool_cache] cache.dir unset — cache disabled")
        enabled_flag = False
    _SETTINGS = _Settings(
        enabled=enabled_flag,
        read=bool(cfg.cache.tool_read),
        write=bool(cfg.cache.tool_write),
        run_dir=run_dir,
        global_dir=str(global_dir) if global_dir else None,
        shards=max(1, int(cfg.cache.tool_shards or 32)),
        blob_max_bytes=max(0, int(cfg.cache.tool_blob_max_bytes)),
        mem_max_bytes=mem_max,
        mem_stripe_max=max(1, mem_max // _MEM_STRIPES),
        queue_max_bytes=max(0, int(cfg.cache.tool_queue_max_bytes)),
    )
    with _STAT_LOCK:
        _HITS.clear()
        _MISSES.clear()
        _DROPPED = 0
    with _QUEUE_BYTES_LOCK:
        _QUEUED_BYTES = 0
    with _STORES_LOCK:
        for store in _STORES.values():
            store.close()
        _STORES.clear()
    if _SETTINGS.enabled:
        mode = ("read" if _SETTINGS.read else "") + ("+write" if _SETTINGS.write else "")
        print(f"[tool_cache] on run={_SETTINGS.run_dir} global={_SETTINGS.global_dir or '-'} "
              f"shards={_SETTINGS.shards} mode={mode.strip('+') or 'off'}")


def enabled() -> bool:
    return _SETTINGS.enabled


class _ShardedStore:
    def __init__(self, namespace: str, run_dir: str, global_dir: Optional[str], shards: int):
        self.namespace = namespace
        self.shards = max(1, int(shards))
        self.run_dir = run_dir
        self._shard_files = [
            os.path.join(self.run_dir, f"{namespace}_shard{idx}.db") for idx in range(self.shards)
        ]
        self._conns = {}
        self._locks = {}
        self._initialized = set()
        self._table_lock = threading.Lock()
        self._global_lock = threading.Lock()
        self._global_conn = None
        if global_dir:
            path = os.path.join(global_dir, f"{namespace}.db")
            if os.path.exists(path):
                try:
                    self._global_conn = sqlite3.connect(
                        f"file:{path}?mode=ro", uri=True, timeout=30.0, check_same_thread=False
                    )
                    self._global_conn.row_factory = sqlite3.Row
                except Exception as exc:
                    print(f"[tool_cache] global tier unavailable for {namespace}: {exc}")

    @staticmethod
    def _ensure_table(conn: sqlite3.Connection) -> None:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_kv (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                meta TEXT NOT NULL DEFAULT '',
                timestamp REAL NOT NULL
            )
            """
        )
        conn.commit()

    def _shard_id(self, key: str) -> int:
        if self.shards == 1:
            return 0
        return int.from_bytes(hashlib.md5(key.encode("utf-8")).digest()[:4], "little") % self.shards

    def _conn(self, shard_id: int, *, initialize: bool) -> sqlite3.Connection:
        with self._table_lock:
            if shard_id not in self._conns:
                if initialize:
                    os.makedirs(self.run_dir, exist_ok=True)
                conn = sqlite3.connect(self._shard_files[shard_id], timeout=30.0, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                self._conns[shard_id] = conn
                self._locks[shard_id] = threading.Lock()
            conn = self._conns[shard_id]
            lock = self._locks[shard_id]
        if initialize and shard_id not in self._initialized:
            with lock:
                if shard_id not in self._initialized:
                    self._ensure_table(conn)
                    self._initialized.add(shard_id)
        return conn

    def get(self, key: str) -> Optional[bytes]:
        shard_id = self._shard_id(key)
        if os.path.exists(self._shard_files[shard_id]):
            try:
                conn = self._conn(shard_id, initialize=False)
                with self._locks[shard_id]:
                    row = conn.execute("SELECT value FROM cache_kv WHERE key = ?", (key,)).fetchone()
                if row is not None:
                    return row["value"]
            except sqlite3.OperationalError as exc:
                if "no such table" not in str(exc).lower():
                    print(f"[tool_cache] {self.namespace} run-tier read failed: {exc}")
            except Exception as exc:
                print(f"[tool_cache] {self.namespace} run-tier read failed: {exc}")
        if self._global_conn is None:
            return None
        try:
            with self._global_lock:
                row = self._global_conn.execute(
                    "SELECT value FROM cache_kv WHERE key = ?", (key,)
                ).fetchone()
            return row["value"] if row is not None else None
        except Exception as exc:
            print(f"[tool_cache] {self.namespace} global-tier read failed: {exc}")
            return None

    def put_many(self, rows) -> None:
        by_shard = {}
        for key, value, meta, stamp in rows:
            by_shard.setdefault(self._shard_id(key), []).append((key, value, meta, stamp))
        for shard_id, batch in by_shard.items():
            try:
                conn = self._conn(shard_id, initialize=True)
                with self._locks[shard_id]:
                    conn.executemany(
                        "INSERT OR REPLACE INTO cache_kv (key, value, meta, timestamp) "
                        "VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    conn.commit()
            except Exception as exc:
                print(f"[tool_cache] {self.namespace} write failed: {exc}")

    def close(self) -> None:
        for conn in list(self._conns.values()):
            try:
                conn.close()
            except Exception:
                pass
        self._conns.clear()
        self._initialized.clear()
        if self._global_conn is not None:
            try:
                self._global_conn.close()
            except Exception:
                pass
            self._global_conn = None


_MEM_STRIPES = 16
_MEM = [OrderedDict() for _ in range(_MEM_STRIPES)]
_MEM_BYTES = [0] * _MEM_STRIPES
_MEM_LOCKS = [threading.Lock() for _ in range(_MEM_STRIPES)]


def _stripe(key: str) -> int:
    if not key:
        return 0
    try:
        return int(key[0], 16)
    except ValueError:
        return ord(key[0]) % _MEM_STRIPES


def _mem_get(namespace: str, key: str) -> Optional[bytes]:
    index = _stripe(key)
    with _MEM_LOCKS[index]:
        entry = _MEM[index].get((namespace, key))
        if entry is not None:
            _MEM[index].move_to_end((namespace, key))
        return entry


def _mem_put(namespace: str, key: str, value: bytes) -> None:
    size = len(value)
    if _SETTINGS.mem_max_bytes <= 0 or size > _SETTINGS.mem_stripe_max:
        return
    index = _stripe(key)
    with _MEM_LOCKS[index]:
        bucket = _MEM[index]
        previous = bucket.pop((namespace, key), None)
        if previous is not None:
            _MEM_BYTES[index] -= len(previous)
        bucket[(namespace, key)] = value
        _MEM_BYTES[index] += size
        while _MEM_BYTES[index] > _SETTINGS.mem_stripe_max and bucket:
            _, evicted = bucket.popitem(last=False)
            _MEM_BYTES[index] -= len(evicted)


def _mem_clear() -> None:
    for index in range(_MEM_STRIPES):
        with _MEM_LOCKS[index]:
            _MEM[index].clear()
            _MEM_BYTES[index] = 0


_QUEUE_MAX = 20000
_BATCH_MAX = 512
_WRITE_QUEUE = None
_WRITER = None
_WRITER_LOCK = threading.Lock()
_WRITER_STOPPING = False
_QUEUE_BYTES_LOCK = threading.Lock()
_QUEUED_BYTES = 0
_DROPPED = 0
_SHUTDOWN = object()


def _release_queue_bytes(batch) -> None:
    global _QUEUED_BYTES
    released = sum(len(item[2]) for item in batch)
    with _QUEUE_BYTES_LOCK:
        _QUEUED_BYTES = max(0, _QUEUED_BYTES - released)


def _writer_loop(write_queue: Queue) -> None:
    while True:
        item = write_queue.get()
        if item is _SHUTDOWN:
            write_queue.task_done()
            return
        batch = [item]
        stop_after_batch = False
        while len(batch) < _BATCH_MAX:
            try:
                nxt = write_queue.get_nowait()
            except Empty:
                break
            if nxt is _SHUTDOWN:
                write_queue.task_done()
                stop_after_batch = True
                break
            batch.append(nxt)
        try:
            _drain(batch)
        finally:
            _release_queue_bytes(batch)
            for _ in batch:
                write_queue.task_done()
        if stop_after_batch:
            return


def _drain(batch) -> None:
    by_namespace = {}
    for namespace, key, value, meta, stamp in batch:
        by_namespace.setdefault(namespace, []).append((key, value, meta, stamp))
    for namespace, rows in by_namespace.items():
        try:
            store = _store(namespace)
            if store is not None:
                store.put_many(rows)
        except Exception as exc:
            print(f"[tool_cache] {namespace} writer failed: {exc}")


def _enqueue(namespace: str, key: str, value: bytes, meta: str) -> None:
    global _WRITE_QUEUE, _WRITER, _WRITER_STOPPING, _DROPPED, _QUEUED_BYTES
    if not (_SETTINGS.enabled and _SETTINGS.write):
        return
    _mem_put(namespace, key, value)
    size = len(value)
    with _QUEUE_BYTES_LOCK:
        if _SETTINGS.queue_max_bytes <= 0 or _QUEUED_BYTES + size > _SETTINGS.queue_max_bytes:
            _DROPPED += 1
            return
        _QUEUED_BYTES += size
    try:
        with _WRITER_LOCK:
            if _WRITER_STOPPING:
                if _WRITER is not None and _WRITER.is_alive():
                    raise Full
                _WRITER = None
                _WRITE_QUEUE = None
                _WRITER_STOPPING = False
            if _WRITER is None or not _WRITER.is_alive():
                _WRITE_QUEUE = Queue(maxsize=_QUEUE_MAX)
                _WRITER = threading.Thread(
                    target=_writer_loop,
                    args=(_WRITE_QUEUE,),
                    name="tool-cache-writer",
                    daemon=True,
                )
                _WRITER.start()
            _WRITE_QUEUE.put_nowait((namespace, key, value, meta, time.time()))
    except Full:
        with _QUEUE_BYTES_LOCK:
            _QUEUED_BYTES = max(0, _QUEUED_BYTES - size)
            _DROPPED += 1
    except Exception:
        with _QUEUE_BYTES_LOCK:
            _QUEUED_BYTES = max(0, _QUEUED_BYTES - size)
            _DROPPED += 1


def flush(timeout: Optional[float] = 30.0) -> bool:
    global _WRITE_QUEUE, _WRITER, _WRITER_STOPPING
    started = time.monotonic()
    with _WRITER_LOCK:
        writer = _WRITER
        write_queue = _WRITE_QUEUE
        if writer is None or write_queue is None or not writer.is_alive():
            _WRITER = None
            _WRITE_QUEUE = None
            _WRITER_STOPPING = False
            return True
        _WRITER_STOPPING = True
        try:
            if timeout is None:
                write_queue.put(_SHUTDOWN)
            else:
                write_queue.put(_SHUTDOWN, timeout=max(0.0, timeout))
        except Full:
            print(f"[tool_cache] flush timed out, {write_queue.qsize()} entries pending")
            return False
    remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
    writer.join(timeout=remaining)
    if writer.is_alive():
        print(f"[tool_cache] flush timed out, {write_queue.qsize()} entries pending")
        return False
    with _WRITER_LOCK:
        if _WRITER is writer:
            _WRITER = None
            _WRITE_QUEUE = None
            _WRITER_STOPPING = False
    return True


def _store(namespace: str) -> Optional[_ShardedStore]:
    if not _SETTINGS.enabled:
        return None
    with _STORES_LOCK:
        if namespace not in _STORES:
            _STORES[namespace] = _ShardedStore(namespace, _SETTINGS.run_dir, _SETTINGS.global_dir, _SETTINGS.shards)
        return _STORES[namespace]


def key_for(namespace: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    version = _KEY_VERSIONS.get(namespace, 1)
    prefix = namespace if version == 1 else f"v{version}\x1f{namespace}"
    return hashlib.sha256(f"{prefix}\x1f{body}".encode("utf-8", "ignore")).hexdigest()


def _safe_meta(meta: str) -> str:
    value = str(meta or "").strip()
    if value.lower().startswith(("http://", "https://")):
        value = sanitize_url(value)
    return value[:1000]


def is_cacheable_text(text: str, status: Optional[int] = None, scope: str = "target") -> bool:
    body = str(text or "")
    meaningful = re.sub(r"(?im)^\s*URL:\s*\S+\s*$", "", body).strip()
    if not meaningful:
        return False
    normalized = meaningful.lstrip().lower()
    if normalized.startswith((
        "[visit] error",
        "[visit] failed",
        "[visit] empty",
        "[search] error",
        "[image_search] no visual matches",
        "[document_parser]",
        "no results found",
        "google search temporarily unavailable",
        "google search timeout",
        "google scholar timeout",
        "jina search timeout",
        "jina search rate-limited",
    )):
        return False
    if "the provided webpage content could not be accessed" in normalized:
        return False
    if re.search(
        r"(?im)^a\s+(?:google\s+)?(?:search|scholar)\b.{0,240}\bfound\s+0\s+results\b",
        meaningful,
    ):
        return False
    try:
        return not is_web_failure(body, status, scope=scope)
    except Exception:
        return False


def _bump(counter: dict, namespace: str) -> None:
    with _STAT_LOCK:
        counter[namespace] = counter.get(namespace, 0) + 1


def _read(namespace: str, key: str) -> Optional[bytes]:
    if not (_SETTINGS.enabled and _SETTINGS.read):
        return None
    hit = _mem_get(namespace, key)
    if hit is not None:
        return hit
    store = _store(namespace)
    if store is None:
        return None
    raw = store.get(key)
    if raw is not None:
        _mem_put(namespace, key, raw)
    return raw


def get_json(namespace: str, key: str):
    if not (_SETTINGS.enabled and _SETTINGS.read):
        return None
    raw = _read(namespace, key)
    if raw is None:
        _bump(_MISSES, namespace)
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception:
        _bump(_MISSES, namespace)
        return None
    _bump(_HITS, namespace)
    return value


def put_json(namespace: str, key: str, value, meta: str = "") -> None:
    if not (_SETTINGS.enabled and _SETTINGS.write):
        return
    try:
        raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
    except Exception as exc:
        print(f"[tool_cache] {namespace} serialize failed: {exc}")
        return
    _enqueue(namespace, key, raw, _safe_meta(meta))


def get_blob(namespace: str, key: str) -> Optional[bytes]:
    if not (_SETTINGS.enabled and _SETTINGS.read):
        return None
    raw = _read(namespace, key)
    _bump(_HITS if raw is not None else _MISSES, namespace)
    return raw


def put_blob(namespace: str, key: str, value: bytes, meta: str = "") -> None:
    if not (_SETTINGS.enabled and _SETTINGS.write) or not value or len(value) > _SETTINGS.blob_max_bytes:
        return
    _enqueue(namespace, key, value, _safe_meta(meta))


def stats() -> dict:
    with _STAT_LOCK:
        namespaces = sorted(set(_HITS) | set(_MISSES))
        return {ns: (_HITS.get(ns, 0), _MISSES.get(ns, 0)) for ns in namespaces}


def log_stats() -> None:
    rows = stats()
    if not rows:
        return
    parts = []
    for namespace, (hit, miss) in rows.items():
        total = hit + miss
        rate = (hit / total * 100.0) if total else 0.0
        parts.append(f"{namespace}={hit}/{total} ({rate:.0f}%)")
    tail = f" dropped={_DROPPED}" if _DROPPED else ""
    print(f"[tool_cache] hits {' '.join(parts)}{tail}")


def _shutdown() -> None:
    drained = flush()
    if not drained:
        log_stats()
        return
    with _STORES_LOCK:
        for store in _STORES.values():
            store.close()
        _STORES.clear()
    log_stats()


atexit.register(_shutdown)






_LEGACY_FAMILIES = {
    "search_cache_merged": (("search_cache", ("query", "result", "timestamp"), "query"),),
    "visit_cache_merged": (
        ("url_content", ("url", "content", "timestamp"), "url"),
        ("url_goal_info", ("url", "goal", "useful_information", "timestamp"), "url, goal"),
    ),
    "scholar_cache_merged": (("scholar_cache", ("query", "result", "timestamp"), "query"),),
}

_LEGACY_DDL = {
    "search_cache": "CREATE TABLE IF NOT EXISTS search_cache (query TEXT PRIMARY KEY, result TEXT NOT NULL, timestamp REAL NOT NULL)",
    "scholar_cache": "CREATE TABLE IF NOT EXISTS scholar_cache (query TEXT PRIMARY KEY, result TEXT NOT NULL, timestamp REAL NOT NULL)",
    "url_content": "CREATE TABLE IF NOT EXISTS url_content (url TEXT PRIMARY KEY, content TEXT NOT NULL, timestamp REAL NOT NULL)",
    "url_goal_info": "CREATE TABLE IF NOT EXISTS url_goal_info (url TEXT NOT NULL, goal TEXT NOT NULL, useful_information TEXT NOT NULL, timestamp REAL NOT NULL, PRIMARY KEY (url, goal))",
}
