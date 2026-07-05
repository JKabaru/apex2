from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidencePolicy:
    max_confidence: float
    label: str


POLICIES: dict[str, EvidencePolicy] = {
    "EXACT": EvidencePolicy(max_confidence=1.0, label="EXACT_MATCH"),
    "ANCHOR": EvidencePolicy(max_confidence=0.7, label="ANCHOR_PROXY"),
    "REGIME": EvidencePolicy(max_confidence=0.5, label="BROAD_REGIME"),
    "COLD_START": EvidencePolicy(max_confidence=0.4, label="COLD_START"),
}


def get_policy(source: str) -> EvidencePolicy:
    return POLICIES.get(source, POLICIES["COLD_START"])
