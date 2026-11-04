from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager


_TOOL_SUBTIMING = threading.local()


@contextmanager
def tool_subphase(name: str):
    bucket = getattr(_TOOL_SUBTIMING, "acc", None)
    if bucket is None:
        bucket = {}
        _TOOL_SUBTIMING.acc = bucket
    started = time.perf_counter()
    try:
        yield
    finally:
        bucket[name] = bucket.get(name, 0.0) + (time.perf_counter() - started)


def drain_tool_subtiming() -> dict:
    bucket = getattr(_TOOL_SUBTIMING, "acc", None)
    if not bucket:
        return {}
    _TOOL_SUBTIMING.acc = {}
    return dict(bucket)


def percentile(values, fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def stage_summary(values, wall_s: float = 0.0) -> dict:
    total_s = sum(values)
    result = {
        "n": len(values),
        "sum_s": round(total_s, 6),
        "mean_s": round(total_s / len(values), 6) if values else 0.0,
        "p50_s": round(percentile(values, 0.50), 6),
        "p95_s": round(percentile(values, 0.95), 6),
        "max_s": round(max(values), 6) if values else 0.0,
    }
    if wall_s > 0:
        result["wall_overlap_x"] = round(total_s / wall_s, 6)
    return result


class TaskTiming:
    def __init__(self, started_at: float | None = None):
        self.started_at = started_at if started_at is not None else time.perf_counter()
        self.stages = {}
        self.stage_counts = {}
        self.metrics = {}
        self.notes = {}
        self.endpoints = Counter()
        self.rounds = []

    def add_stage(self, name: str, seconds: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + max(0.0, float(seconds))
        self.stage_counts[name] = self.stage_counts.get(name, 0) + 1

    @contextmanager
    def span(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_stage(name, time.perf_counter() - started)

    def increment(self, name: str, amount: int = 1) -> None:
        self.metrics[name] = int(self.metrics.get(name, 0)) + int(amount)

    def add_metric(self, name: str, amount: float) -> None:
        self.metrics[name] = float(self.metrics.get(name, 0.0)) + float(amount)

    def set_metric(self, name: str, value) -> None:
        self.metrics[name] = value

    def max_metric(self, name: str, value: float) -> None:
        previous = self.metrics.get(name)
        self.metrics[name] = value if previous is None else max(previous, value)

    def note(self, name: str, value) -> None:
        if value is not None:
            self.notes[name] = str(value)

    def record_endpoint(self, host: str, port: int) -> None:
        self.endpoints[f"{host}:{port}"] += 1

    def add_round(self, row: dict) -> None:
        self.rounds.append(row)

    def checkpoint(self) -> dict:
        return {
            "stages_s": dict(self.stages),
            "stage_counts": dict(self.stage_counts),
            "metrics": dict(self.metrics),
            "endpoints": dict(self.endpoints),
        }

    def delta_since(self, checkpoint: dict) -> dict:
        stages_before = checkpoint.get("stages_s") or {}
        counts_before = checkpoint.get("stage_counts") or {}
        metrics_before = checkpoint.get("metrics") or {}
        endpoints_before = checkpoint.get("endpoints") or {}

        stages = {
            name: round(value - float(stages_before.get(name, 0.0)), 6)
            for name, value in self.stages.items()
            if value - float(stages_before.get(name, 0.0)) > 0
        }
        stage_counts = {
            name: count - int(counts_before.get(name, 0))
            for name, count in self.stage_counts.items()
            if count - int(counts_before.get(name, 0)) > 0
        }
        metrics = {}
        for name, value in self.metrics.items():
            previous = metrics_before.get(name, 0)
            if name.startswith("context.max_"):
                metrics[name] = value
                continue
            if isinstance(value, (int, float)) and isinstance(previous, (int, float)):
                delta = value - previous
                if delta:
                    metrics[name] = delta
            elif value != previous:
                metrics[name] = value
        endpoints = {
            name: count - int(endpoints_before.get(name, 0))
            for name, count in self.endpoints.items()
            if count - int(endpoints_before.get(name, 0)) > 0
        }
        return {
            "stages_s": stages,
            "stage_counts": stage_counts,
            "metrics": metrics,
            "endpoints": endpoints,
        }

    def snapshot(self) -> dict:
        return {
            "stages_s": {name: round(value, 6) for name, value in self.stages.items()},
            "stage_counts": dict(self.stage_counts),
            "metrics": dict(self.metrics),
            "notes": dict(self.notes),
            "endpoints": dict(self.endpoints),
            "rounds": list(self.rounds),
        }


def add_snapshot_stage(snapshot: dict, name: str, seconds: float) -> None:
    stages = snapshot.setdefault("stages_s", {})
    counts = snapshot.setdefault("stage_counts", {})
    stages[name] = round(float(stages.get(name, 0.0)) + max(0.0, float(seconds)), 6)
    counts[name] = int(counts.get(name, 0)) + 1


class TimingAccumulator:
    def __init__(self):
        self.completed = 0
        self.status = Counter()
        self.stage_samples = defaultdict(list)
        self.metric_sums = defaultdict(float)
        self.endpoints = Counter()

    def add(self, row: dict) -> None:
        self.completed += 1
        self.status[row.get("status", "unknown")] += 1
        for name, seconds in (row.get("stages_s") or {}).items():
            self.stage_samples[name].append(float(seconds))
        for name, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                self.metric_sums[name] += float(value)
        self.endpoints.update(row.get("endpoints") or {})

    def summary(self, wall_s: float) -> dict:
        llm_api_s = sum(self.stage_samples.get("llm.api_request", ()))
        return {
            "completed": self.completed,
            "status": dict(self.status),
            "wall_s": round(wall_s, 6),
            "throughput_qps": round(self.completed / wall_s, 6) if wall_s > 0 else 0.0,
            "llm_api_parallelism": round(llm_api_s / wall_s, 6) if wall_s > 0 else 0.0,
            "stages": {
                name: stage_summary(values, wall_s)
                for name, values in self.stage_samples.items()
            },
            "metric_sums": dict(self.metric_sums),
            "endpoints": dict(self.endpoints),
        }
