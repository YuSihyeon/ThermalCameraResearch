from __future__ import annotations

from collections import deque

from .model import Sample


class Synchronizer:
    """One-to-one nearest *available* matching, never a hardware sync claim.

    Input order must increase within each stream. A bounded queue intentionally
    drops old samples. A bundle is emitted as soon as all required streams have
    a candidate; it does not wait for a potentially better future candidate.
    """

    def __init__(self, anchor, required, tolerance_ns, capacity=32):
        if anchor not in required or not required or len(set(required)) != len(required):
            raise ValueError("required streams must be unique and include anchor")
        if type(tolerance_ns) is not int or tolerance_ns < 0 or type(capacity) is not int or capacity < 1:
            raise ValueError("tolerance_ns and capacity must be nonnegative/positive integers")
        self.anchor, self.tolerance_ns, self.capacity = anchor, tolerance_ns, capacity
        self.queues = {name: deque() for name in required}
        self.last = {}
        self.clock = None
        self.dropped = 0

    def push(self, sample: Sample):
        if sample.stream not in self.queues:
            return []
        if self.clock is not None and sample.clock != self.clock:
            raise ValueError("cannot synchronize different clock domains; map device clocks first")
        if sample.timestamp_ns <= self.last.get(sample.stream, -1):
            raise ValueError("timestamps must strictly increase within each stream")
        self.clock = sample.clock
        self.last[sample.stream] = sample.timestamp_ns
        queue = self.queues[sample.stream]
        if len(queue) == self.capacity:
            queue.popleft()
            self.dropped += 1
        queue.append(sample)
        bundles = []
        anchors = self.queues[self.anchor]
        while anchors:
            anchor = anchors[0]
            chosen = {self.anchor: anchor}
            impossible = False
            for name, candidates in self.queues.items():
                if name == self.anchor:
                    continue
                while candidates and candidates[0].timestamp_ns < anchor.timestamp_ns - self.tolerance_ns:
                    candidates.popleft()
                    self.dropped += 1
                if candidates:
                    best = min(candidates, key=lambda p: abs(p.timestamp_ns - anchor.timestamp_ns))
                    if abs(best.timestamp_ns - anchor.timestamp_ns) <= self.tolerance_ns:
                        chosen[name] = best
                    elif candidates[0].timestamp_ns > anchor.timestamp_ns + self.tolerance_ns:
                        impossible = True
            if len(chosen) == len(self.queues):
                for name, selected in chosen.items():
                    while self.queues[name][0] is not selected:
                        self.queues[name].popleft()
                        self.dropped += 1
                    self.queues[name].popleft()
                bundles.append(chosen)
            elif impossible:
                anchors.popleft()
                self.dropped += 1
            else:
                break
        return bundles
