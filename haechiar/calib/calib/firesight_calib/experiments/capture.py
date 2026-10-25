from __future__ import annotations

import importlib
import math
import multiprocessing
import queue
import time
import uuid

from .model import Sample
from .config import MODES
from .session import SessionWriter


def _worker(spec, clock, messages, stop, dropped):
    adapter = None
    try:
        module, name = spec["factory"].split(":")
        adapter = getattr(importlib.import_module(module), name)(options=spec.get("options", {}), clock=clock)
        if not callable(getattr(adapter, "read", None)) or not callable(getattr(adapter, "close", None)):
            raise ValueError("source adapter needs read(timeout_s) and close()")
        while not stop.is_set():
            samples = adapter.read(0.1)
            if samples is None:
                continue
            if isinstance(samples, Sample):
                samples = [samples]
            for sample in samples:
                if not isinstance(sample, Sample) or sample.clock != clock or sample.provenance != "measured":
                    raise ValueError("live adapters require measured Sample objects on the provided host clock")
                try:
                    messages.put(("sample", sample), timeout=0.01)
                except queue.Full:
                    with dropped.get_lock():
                        dropped.value += 1
    except BaseException as exc:
        while not stop.is_set():
            try:
                messages.put(("error", f"{spec['factory']}: {exc}"), timeout=0.1)
                break
            except queue.Full:
                continue
    finally:
        if adapter is not None:
            adapter.close()


def capture_sources(sources, output, metadata, duration_s=60., max_samples=0):
    """Separate processes bound even a hung vendor open/read/close operation."""
    if not sources or not math.isfinite(duration_s) or duration_s <= 0:
        raise ValueError("sources and positive finite duration_s are required")
    if type(max_samples) is not int or max_samples < 0:
        raise ValueError("max_samples must be a nonnegative integer")
    for spec in sources:
        if not isinstance(spec.get("factory"), str) or spec["factory"].count(":") != 1:
            raise ValueError("source factory must be module:factory")
    context = multiprocessing.get_context("spawn")
    messages, stop = context.Queue(maxsize=128), context.Event()
    dropped = context.Value("i", 0)
    clock = "host_monotonic:" + uuid.uuid4().hex
    workers = []
    count, terminated = 0, 0
    stream_counts = {}
    try:
        with SessionWriter(output, {**metadata, "sources": sources, "duration_s": duration_s,
                                    "timestamp_policy": "host receipt; not hardware synchronized"}) as writer:
            try:
                for spec in sources:
                    process = context.Process(target=_worker, args=(spec, clock, messages, stop, dropped))
                    process.start()
                    workers.append(process)
                deadline = time.monotonic() + duration_s
                while time.monotonic() < deadline and (max_samples == 0 or count < max_samples):
                    try:
                        kind, value = messages.get(timeout=min(0.1, max(0.001, deadline - time.monotonic())))
                    except queue.Empty:
                        if any(not worker.is_alive() for worker in workers):
                            raise RuntimeError("source worker exited unexpectedly")
                        continue
                    if kind == "error":
                        raise RuntimeError(value)
                    writer.append(value)
                    stream_counts[value.stream] = stream_counts.get(value.stream, 0) + 1
                    count += 1
                if count == 0:
                    raise RuntimeError("no samples received before capture deadline")
                missing = set(MODES.get(metadata.get("mode"), metadata.get("streams", []))) - stream_counts.keys()
                if missing:
                    raise RuntimeError("required streams produced no samples: " + ", ".join(sorted(missing)))
            finally:
                stop.set()
                for worker in workers:
                    worker.join(0.4)
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(0.4)
                        terminated += 1
                    if worker.is_alive():
                        worker.kill()
                        worker.join(0.4)
                writer.document.update(queue_dropped_samples=dropped.value, terminated_workers=terminated,
                                       stream_counts=stream_counts)
    finally:
        messages.close()
        messages.join_thread()
    return {"samples": count, "output": str(output), "queue_dropped_samples": dropped.value,
            "terminated_workers": terminated, "hardware_verified": False}
