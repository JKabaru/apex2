from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvidencePolicy:
    max_confidence: float
    label: str


DEFAULT_POLICIES: dict[str, EvidencePolicy] = {
    "EXACT": EvidencePolicy(max_confidence=1.0, label="EXACT_MATCH"),
    "ANCHOR": EvidencePolicy(max_confidence=0.7, label="ANCHOR_PROXY"),
    "REGIME": EvidencePolicy(max_confidence=0.5, label="BROAD_REGIME"),
    "COLD_START": EvidencePolicy(max_confidence=0.4, label="COLD_START"),
}

POLICIES: dict[str, EvidencePolicy] = dict(DEFAULT_POLICIES)


def get_policy(source: str) -> EvidencePolicy:
    return POLICIES.get(source, POLICIES["COLD_START"])


def configure_from_config(config: dict | None = None) -> None:
    policies = dict(DEFAULT_POLICIES)
    if config:
        evidence_cfg = config.get("evidence", {})
        for tier_key, policy in DEFAULT_POLICIES.items():
            tier_cfg_key = tier_key.lower()
            tier_config = evidence_cfg.get(tier_cfg_key, {})
            max_conf = tier_config.get("max_confidence")
            if max_conf is not None:
                policies[tier_key] = EvidencePolicy(
                    max_confidence=float(max_conf),
                    label=policy.label,
                )
    POLICIES.clear()
    POLICIES.update(policies)
