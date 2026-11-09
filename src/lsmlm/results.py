# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
# Portions derived from third-party MIT-licensed code; see THIRD_PARTY_NOTICES.md.
"""Result container for :func:`lsmlm.lsmlm`."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["LSMLMError", "LSMLMResult"]

_RULE = "-" * 66


class LSMLMError(ValueError):
    """Raised for invalid options or a configuration that cannot be estimated."""


def _format_time(value: Any) -> str:
    """Format a time value the way Stata prints an integer-valued time variable."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


@dataclass(frozen=True)
class LSMLMResult:
    """Outcome of a Lee-Strazicich minimum LM unit root test.

    The attribute names mirror the ``r()`` results of the Stata command
    ``lsmlm`` so that code can be translated one for one. Break-specific
    attributes are ``None`` when they do not apply (for example ``tb2`` with
    one break, or ``tD1`` in the crash model).

    Attributes
    ----------
    tau
        The test statistic: the minimum t-ratio on ``S(t-1)`` over the break
        search.
    k
        Number of augmenting lags chosen by the general-to-specific rule.
    breaks
        Number of breaks (0, 1 or 2).
    N
        Number of observations in the final test regression.
    maxlag
        Maximum lag considered.
    trim
        Trimming fraction of the break search.
    cv1, cv5, cv10
        Interpolated 1%, 5% and 10% critical values.
    reject1, reject5, reject10
        Whether the unit root null is rejected at each level.
    siglevel
        ``"1%"``, ``"5%"``, ``"10%"`` or ``"Not significant"``.
    model
        ``"break"`` or ``"crash"``.
    varname
        Label of the series (display only).
    tb1, tb2
        Break dates, in the units of the time variable.
    tb1_index, tb2_index
        One-based positions of the breaks in the (sorted, non-missing) sample.
    lambda1, lambda2
        Break fractions used for the critical-value lookup.
    break1_sig10, break2_sig10
        Whether the break dummies are relevant (largest ``|t|`` >= 1.645).
    tB1, tB2
        t-ratios on the one-period (level) break dummies.
    tD1, tD2
        t-ratios on the step (trend) break dummies; trend-break model only.
    mindist
        Minimum distance between the two breaks (two-break case only).
    tmin, tmax
        First and last time value of the sample.
    """

    tau: float
    k: int
    breaks: int
    N: int
    maxlag: int
    trim: float
    cv1: float
    cv5: float
    cv10: float
    reject1: bool
    reject5: bool
    reject10: bool
    siglevel: str
    model: str
    varname: str
    tb1: Any = None
    tb1_index: int | None = None
    lambda1: float | None = None
    break1_sig10: bool | None = None
    tB1: float | None = None
    tD1: float | None = None
    tb2: Any = None
    tb2_index: int | None = None
    lambda2: float | None = None
    break2_sig10: bool | None = None
    tB2: float | None = None
    tD2: float | None = None
    mindist: int | None = None
    tmin: Any = None
    tmax: Any = None

    @property
    def decision(self) -> str:
        """Plain-language outcome at the 10% level."""
        if self.reject10:
            return f"Reject unit root at {self.siglevel}"
        return "Do not reject unit root"

    def to_dict(self) -> dict[str, Any]:
        """Return all results as a plain dictionary (handy for building tables)."""
        return asdict(self)

    def summary(self) -> str:
        """Return the results table, laid out like the Stata command's output."""
        if self.breaks == 0:
            description = "no break (Schmidt-Phillips)"
        elif self.model == "crash":
            description = "crash (level shift)"
        else:
            description = "trend break (level + trend shift)"

        lines = [
            _RULE,
            "  Lee-Strazicich minimum LM unit root test",
            _RULE,
            f"  Series:        {self.varname}",
            f"  Model:         {description}",
            f"  Breaks:        {self.breaks}   (endogenously selected)",
            f"  Max lag:       {self.maxlag}    Lags chosen (GTOS): {self.k}",
            f"  Obs (regr.):   {self.N}",
            f"  Sample:        {_format_time(self.tmin)} to {_format_time(self.tmax)}",
        ]
        if self.breaks >= 1:
            lines.append(
                f"  Break 1:       {_format_time(self.tb1)}   (lambda1 = {self.lambda1:5.3f})"
            )
        if self.breaks == 2:
            lines.append(
                f"  Break 2:       {_format_time(self.tb2)}   (lambda2 = {self.lambda2:5.3f})"
            )
        lines += [
            _RULE,
            f"  Test statistic (t on S(t-1)):  {self.tau:9.4f}",
            f"  Critical values:   1% {self.cv1:7.3f}    5% {self.cv5:7.3f}"
            f"    10% {self.cv10:7.3f}",
            f"  Decision:      {self.decision}",
            _RULE,
        ]
        if self.breaks >= 1:
            lines.append("  Break significance (|t| on dummies >= 1.645):")
            lines.append(f"    Break 1:     {_significance(self.break1_sig10)}")
            if self.breaks == 2:
                lines.append(f"    Break 2:     {_significance(self.break2_sig10)}")
            lines.append(_RULE)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _significance(flag: bool | None) -> str:
    return "significant" if flag else "not significant"
