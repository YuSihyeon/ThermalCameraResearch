import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "edge/orin"))
from edge_filter import EdgeFilterConfig, select_edges  # noqa: E402

checks = []


def selected(mask, **kwargs):
    frame = np.zeros((*mask.shape, 3), np.uint8)
    return select_edges(mask.astype(np.float32), frame,
                        EdgeFilterConfig(gradient_high=0, gradient_low=0, **kwargs))


def check(name, passed):
    if not passed:
        raise AssertionError(name)
    checks.append(name)


def main():
    base = np.zeros((64, 64), bool)
    base[30, 5:55] = True
    check("long thin horizontal survives", np.array_equal(selected(base).mask > 0, base))
    diagonal = np.eye(64, dtype=bool)
    check("long diagonal survives", np.array_equal(selected(diagonal).mask > 0, diagonal))
    corner = base.copy()
    corner[10:31, 5] = True
    check("L corner is not a spur", np.array_equal(selected(corner).mask > 0, corner))
    branched = base.copy()
    branched[26:30, 30] = True
    old = selected(branched, min_branch_length=0, min_extent=0)
    new = selected(branched)
    check("old minimum length keeps attached spur", np.array_equal(old.mask > 0, branched))
    check("short spur removed with trunk unchanged", np.array_equal(new.mask > 0, base))
    check("spur metric counts actual removal", new.metrics["pruned_branch_pixels"] == 4)
    branched[20:30, 40] = True
    branch_result = selected(branched).mask > 0
    check("long branch survives", branch_result[20:30, 40].all())
    loop = np.zeros((64, 64), bool)
    loop[10, 10:15] = True
    loop[14, 10:15] = True
    loop[10:15, 10] = True
    loop[10:15, 14] = True
    check("small loop previously passes length", np.any(selected(loop, min_branch_length=0, min_extent=0).mask))
    check("small compact loop removed", not np.any(selected(loop).mask))
    large_loop = np.zeros((64, 64), bool)
    large_loop[10, 10:40] = large_loop[39, 10:40] = True
    large_loop[10:40, 10] = large_loop[10:40, 39] = True
    check("large closed structure survives", np.any(selected(large_loop).mask))
    exact = base.copy()
    exact[24:30, 30] = True
    check("branch at exact threshold survives", selected(exact).mask[24:30, 30].all())
    border = np.zeros((64, 64), bool)
    border[0, :] = True
    check("border edge survives", selected(border).mask[0].all())
    check("empty frame stays empty", not np.any(selected(np.zeros((64, 64), bool)).mask))
    for key in ["min_branch_length", "min_extent"]:
        for value in [-1, float("nan"), float("inf")]:
            try:
                EdgeFilterConfig(**{key: value})
            except ValueError:
                checks.append(f"reject {key}={value}")
            else:
                raise AssertionError(f"accepted invalid {key}")
    print(f"PASS: {len(checks)} geometric edge cleanup contracts")
    return checks


if __name__ == "__main__":
    main()
