from __future__ import annotations

import enum


class RetrievalScope(str, enum.Enum):
    EXACT = "EXACT"
    ANCHOR = "ANCHOR"
    REGIME = "REGIME"
