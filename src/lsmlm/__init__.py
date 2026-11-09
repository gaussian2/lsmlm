# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Junsoo Lee
"""lsmlm: Lee-Strazicich minimum LM unit root test with zero, one or two breaks.

Python package by Junsoo Lee (University of Alabama); a port of the Stata
command ``lsmlm`` by Ibrahim Ongoren (Pamukkale University). The main entry
point is :func:`lsmlm.lsmlm`.
"""

from .core import lsmlm
from .critical_values import CriticalValues, critical_values
from .results import LSMLMError, LSMLMResult

__all__ = [
    "CriticalValues",
    "LSMLMError",
    "LSMLMResult",
    "__version__",
    "critical_values",
    "lsmlm",
]

__version__ = "1.2.0"

#: Version of the Stata command (lsmlm.ado) that this package reproduces.
STATA_REFERENCE_VERSION = "1.1.0"
