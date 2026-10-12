"""`budgetgs fit`: land a checkpoint on a byte budget by bisecting block precision.

Decision B of the Week-1 plan: the controller has a single actuator, the
global block precision (``levels``).  Each bisection step really writes the
serialized file and reads the byte count back, so the measured size, not a
prediction, is the feedback signal.  Landing tolerance is
``max(1 % of budget, 256 KiB)``.

``bytes(levels)`` is monotone non-decreasing in ``levels`` (more quantisation
levels -> more distinct symbols -> longer range-coder output), so a plain
bisection finds the largest ``levels`` whose real size stays under the
budget, then compares it with the next level for closeness.
"""

from __future__ import annotations

import os
import tempfile

from ..modules.serialize import write_container


class FitResult:
    """Outcome of one ``fit`` run."""

    __slots__ = (
        "levels", "bytes", "budget", "error_bytes", "error_frac",
        "evals", "met_tolerance", "status",
    )

    def __init__(
        self,
        levels: int,
        bytes_: int,
        budget: int,
        error_bytes: int,
        evals: int,
        met_tolerance: bool,
        status: str,
    ) -> None:
        self.levels = levels
        self.bytes = bytes_
        self.budget = budget
        self.error_bytes = error_bytes
        self.error_frac = error_bytes / budget if budget else 0.0
        self.evals = evals
        self.met_tolerance = met_tolerance
        self.status = status

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FitResult(levels={self.levels}, bytes={self.bytes}, "
            f"budget={self.budget}, err={self.error_bytes}B "
            f"({self.error_frac * 100:.3f}%), evals={self.evals}, "
            f"met_tolerance={self.met_tolerance}, status={self.status!r})"
        )


def _real_bytes(attributes: dict[str, list], levels: int, workdir: str) -> int:
    """Write the container once and return its actual size in bytes."""
    fd, path = tempfile.mkstemp(suffix=".bin", dir=workdir)
    os.close(fd)
    try:
        return write_container(path, attributes, levels=levels)
    finally:
        os.remove(path)


def fit_levels(
    attributes: dict[str, list],
    budget: int,
    levels_min: int = 2,
    levels_max: int = 8192,
    tolerance: float = 0.01,
    workdir: str | None = None,
) -> FitResult:
    """Bisect the global block precision so the real size hits ``budget``.

    Parameters
    ----------
    attributes : flat dict of the six attribute groups (ledger keys).
    budget : target size in bytes (10^6 bytes per MB, per plan conventions).
    levels_min / levels_max : bisection range for the global block precision.
    tolerance : relative landing tolerance; the absolute tolerance is
        ``max(tolerance * budget, 256 * 1024)``.
    workdir : directory for temporary files (defaults to the system temp dir).

    Returns
    -------
    FitResult with the chosen ``levels``, the real byte count, the signed
    error and whether ``max(1 %, 256 KiB)`` was met.
    """
    if levels_min < 2:
        raise ValueError("levels_min must be >= 2 (at least 2 quantisation levels)")
    if levels_max < levels_min:
        raise ValueError("levels_max must be >= levels_min")
    if budget <= 0:
        raise ValueError("budget must be positive")

    eps = max(tolerance * budget, 256 * 1024)

    with tempfile.TemporaryDirectory(prefix="budgetgs-fit-", dir=workdir) as td:
        # Probe the extremes first: the budget may lie outside the reachable range.
        b_min = _real_bytes(attributes, levels_min, td)
        if b_min > budget:
            return FitResult(
                levels_min, b_min, budget, b_min - budget, 1, False,
                "over-budget at minimum precision",
            )

        b_max = _real_bytes(attributes, levels_max, td)
        if b_max <= budget:
            # Everything fits even at maximum precision; landing is levels_max.
            err = budget - b_max
            met = err <= eps
            return FitResult(
                levels_max, b_max, budget, err, 2, met,
                "fits at max precision" if met else "max precision below budget (tolerance missed)",
            )

        # Invariant: bytes(lo) <= budget < bytes(hi).
        lo, hi = levels_min, levels_max
        evals = 2
        while hi - lo > 1:
            mid = (lo + hi) // 2
            evals += 1
            if _real_bytes(attributes, mid, td) <= budget:
                lo = mid
            else:
                hi = mid

        # lo is the largest level whose size stays under the budget; pick the
        # closer of lo and hi.
        b_lo = _real_bytes(attributes, lo, td)
        b_hi = _real_bytes(attributes, hi, td)
        evals += 2
        err_lo = budget - b_lo
        err_hi = b_hi - budget
        if err_lo <= err_hi:
            chosen, bytes_, err = lo, b_lo, err_lo
        else:
            chosen, bytes_, err = hi, b_hi, err_hi

        return FitResult(
            chosen, bytes_, budget, err, evals, err <= eps,
            "tolerance met" if err <= eps else
            "tolerance missed (neighbouring levels straddle the budget)",
        )
