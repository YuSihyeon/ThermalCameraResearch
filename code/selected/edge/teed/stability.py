"""Predeclared stability criteria, independent of hardware and inference."""
import math
import numpy as np


def distribution(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {'mean': None, 'p50': None, 'p95': None, 'p99': None, 'max': None}
    return dict(zip(('mean', 'p50', 'p95', 'p99', 'max'),
                    [float(array.mean())] + list(map(float, np.percentile(array, (50, 95, 99)))) + [float(array.max())]))


def assess_stability(rows, target_fps, started_s, ended_s, captured, dropped,
                     minimum_duration_s=180.):
    if not all(math.isfinite(v) for v in (target_fps, started_s, ended_s, minimum_duration_s)):
        raise ValueError('stability clock and criteria must be finite')
    if target_fps <= 0 or ended_s <= started_s or minimum_duration_s <= 0 or captured < 0 or dropped < 0:
        raise ValueError('invalid stability interval or counts')
    elapsed = ended_s - started_s
    budget = 1000. / target_fps
    sequence, prior_output = -1, started_s
    for row in rows:
        numeric = (row['output_s'], row['processing_ms'], row['frame_age_ms'])
        if not all(math.isfinite(v) for v in numeric) or row['processing_ms'] < 0 or row['frame_age_ms'] < 0:
            raise ValueError('nonfinite or negative measured latency')
        if row['sequence'] <= sequence or row['output_s'] < prior_output or row['output_s'] > ended_s:
            raise ValueError('nonmonotonic or out-of-run output')
        sequence, prior_output = row['sequence'], row['output_s']
    processing = distribution([r['processing_ms'] for r in rows])
    age = distribution([r['frame_age_ms'] for r in rows])
    outputs = np.asarray([r['output_s'] for r in rows])
    # Boundary stalls count too: a final 200ms freeze must not disappear just
    # because there was no next output from which to compute an interval.
    gaps = distribution(np.diff(np.concatenate(([started_s], outputs, [ended_s]))) * 1000.)
    fps = len(rows) / elapsed
    windows = []
    for i in range(int(elapsed / 10.)):
        begin = started_s + 10. * i
        count = int(np.count_nonzero((outputs >= begin) & (outputs < begin + 10.)))
        windows.append({'start_s': 10. * i, 'frames': count, 'fps': count / 10.})
    miss_fraction = (sum(r['processing_ms'] > budget for r in rows) / float(len(rows))) if rows else 1.
    drop_fraction = dropped / float(max(captured, 1))
    checks = {
        'qualified_duration': elapsed >= minimum_duration_s,
        'nonempty': bool(rows),
        'mean_fps_at_least_99pct': fps >= .99 * target_fps,
        'all_10s_windows_at_least_98pct': bool(windows) and all(w['fps'] >= .98 * target_fps for w in windows),
        'processing_p99_within_frame_budget': processing['p99'] is not None and processing['p99'] <= budget,
        'processing_deadline_misses_at_most_1pct': miss_fraction <= .01,
        'capture_drops_at_most_1pct': drop_fraction <= .01,
        'host_frame_age_p99_within_2_frames': age['p99'] is not None and age['p99'] <= 2. * budget,
        'output_gap_p99_within_1_5_frames': gaps['p99'] is not None and gaps['p99'] <= 1.5 * budget,
        'max_output_gap_within_3_frames': gaps['max'] is not None and gaps['max'] <= 3. * budget,
    }
    return {'passed': all(checks.values()), 'checks': checks, 'target_fps': target_fps,
            'minimum_duration_s': minimum_duration_s, 'elapsed_s': elapsed, 'processed': len(rows),
            'effective_fps': fps, 'frame_budget_ms': budget,
            'processing_ms': processing, 'host_frame_age_ms': age, 'output_gap_ms': gaps,
            'deadline_miss_pct': miss_fraction * 100., 'drop_pct': drop_fraction * 100.,
            'depth_valid_pct': 100. * sum(bool(r['depth_valid']) for r in rows) / max(len(rows), 1),
            'windows': windows,
            'latency_scope': 'host capture read completion to output callback completion; excludes sensor exposure/driver buffer and physical display scanout'}
