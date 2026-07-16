from __future__ import annotations

import hashlib
import json

from pydantic import BaseModel

from src.intelligence.models import EvidenceProvenance, ExperienceEvidence, PromptContext
from src.intelligence import templates as tpl


class ExperienceEvidenceFormatter:
    """Linguistic translator — the ONLY class allowed to generate strings.

    Maps structured ExperienceEvidence data into domain-language text
    using explicit templates from templates.py.
    """

    def __init__(self) -> None:
        self._version = "1.0"

    def format(self, evidence: ExperienceEvidence) -> PromptContext:
        sections: list[str] = []
        section_order: list[str] = []

        sections.append(tpl.EVIDENCE_HEADER)
        section_order.append("header")

        # metadata
        sections.append(
            f"Matched historical opportunities: {evidence.sample_size} "
            f"| Quality: {evidence.evidence_quality}"
        )
        section_order.append("metadata")

        # episode / intra-trade trajectory
        if evidence.records_with_episodes > 0:
            sections.append(tpl.EPISODE_TEMPLATE.format(
                avg=evidence.avg_episode_count,
                total=evidence.total_episodes,
                count=evidence.records_with_episodes,
            ))
            section_order.append("episodes")

        # noise warning for statistically insignificant samples
        if evidence.sample_size < 10:
            sections.append(
                "--- STATISTICAL NOISE WARNING ---\n"
                f"The historical sample size (N={evidence.sample_size}) is too small to be "
                "statistically significant.\n"
                "The reported win rate and PnL are STATISTICAL NOISE, not ground truth.\n"
                "YOU MUST IGNORE THESE NUMBERS. Do not let a tiny sample size force you to ABSTAIN.\n"
                "Base your decision entirely on the real-time Scanner signal and current Market Context."
            )
            section_order.append("noise_warning")

        # outcomes
        if evidence.win_rate_pct is not None and evidence.median_pnl_atr is not None:
            conf_label = self._confidence_label(evidence.overall_confidence)
            win_str = tpl.WINRATE_TEMPLATE.format(
                pct=round(evidence.win_rate_pct, 1),
                count=evidence.sample_size,
                total=evidence.sample_size,
            )
            sections.append(
                f"Expected outcome: Median PnL "
                f"{round(evidence.median_pnl_atr, 2)} ATR "
                f"| {win_str} | Confidence: {conf_label}"
            )
            section_order.append("outcomes")

        # risk — adverse movement
        if evidence.median_mae_atr is not None:
            sections.append(tpl.MAE_TEMPLATE.format(
                value=round(evidence.median_mae_atr, 2),
                iqr=round(evidence.pnl_iqr, 2) if evidence.pnl_iqr else "N/A",
            ))
            section_order.append("risk_mae")

        # risk — favorable movement
        if evidence.median_mfe_atr is not None:
            sections.append(tpl.MFE_TEMPLATE.format(
                value=round(evidence.median_mfe_atr, 2),
                iqr=round(evidence.pnl_iqr, 2) if evidence.pnl_iqr else "N/A",
            ))
            section_order.append("risk_mfe")

        # timing
        if evidence.median_duration_bars is not None:
            sections.append(tpl.TIMING_TEMPLATE.format(
                value=round(evidence.median_duration_bars),
                iqr=round(evidence.duration_iqr, 1) if evidence.duration_iqr else "N/A",
            ))
            section_order.append("timing")

        # patterns — success
        for pat in evidence.success_patterns:
            conf_label = self._confidence_label(pat.confidence_score)
            sections.append(tpl.PATTERN_TEMPLATE.format(
                field=pat.field,
                value=pat.value,
                frequency=round(pat.frequency * 100, 1),
                confidence=conf_label,
            ))
            section_order.append(f"pattern_success_{pat.field}_{pat.value}")

        # patterns — failure
        for pat in evidence.failure_patterns:
            conf_label = self._confidence_label(pat.confidence_score)
            sections.append(tpl.PATTERN_TEMPLATE.format(
                field=pat.field,
                value=pat.value,
                frequency=round(pat.frequency * 100, 1),
                confidence=conf_label,
            ))
            section_order.append(f"pattern_failure_{pat.field}_{pat.value}")

        # bias summary
        total = evidence.sample_size or 1
        for symbol, count in evidence.bias_summary.symbol_distribution.items():
            pct = round(count / total * 100, 1)
            sections.append(tpl.BIAS_TEMPLATE.format(value=symbol, pct=pct))
            section_order.append(f"bias_{symbol}")
        for tf, count in evidence.bias_summary.timeframe_distribution.items():
            pct = round(count / total * 100, 1)
            sections.append(tpl.BIAS_TEMPLATE.format(value=tf, pct=pct))
            section_order.append(f"bias_{tf}")

        # representatives
        if evidence.representatives:
            ids = ", ".join(r.experience_id for r in evidence.representatives)
            sections.append(tpl.REPRESENTATIVES_TEMPLATE.format(ids=ids))
            section_order.append("representatives")

        context_str = "\n".join(sections)

        return PromptContext(
            context_string=context_str,
            section_order=section_order,
            template_version=self._version,
            source_evidence_hash=_compute_hash(evidence),
            token_count=len(context_str.split()),
        )

    @staticmethod
    def _confidence_label(score: float) -> str:
        if score >= 0.8:
            return "HIGH"
        if score >= 0.5:
            return "MEDIUM"
        return "LOW"


def _compute_hash(model: BaseModel) -> str:
    dump = json.dumps(model.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(dump.encode()).hexdigest()[:16]
