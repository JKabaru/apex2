from __future__ import annotations

from typing import Optional

import structlog

from src.evaluation.models import DecisionCapture, DecisionEvaluation
from src.models.learning.trade_experience import LearningManifest

logger = structlog.get_logger("decision_evaluation_engine")


class DecisionEvaluationEngine:
    """Pure function. Same inputs always produce the same DecisionEvaluation.
    No I/O, no randomness, no hidden state."""

    VERSION = "4.7.0"

    def evaluate(
        self,
        manifest: LearningManifest,
        capture: DecisionCapture,
        actual_side: str,
        actual_quantity: float,
        actual_exit_reason: Optional[str],
    ) -> DecisionEvaluation | None:
        try:
            return self._evaluate(manifest, capture, actual_side, actual_quantity, actual_exit_reason)
        except Exception as e:
            logger.error(
                "DecisionEvaluationEngine failed",
                opportunity_id=capture.opportunity_id,
                error=str(e),
            )
            return None

    def _evaluate(
        self,
        manifest: LearningManifest,
        capture: DecisionCapture,
        actual_side: str,
        actual_quantity: float,
        actual_exit_reason: Optional[str],
    ) -> DecisionEvaluation:
        # ── Extract outcome data from manifest ──
        exp = manifest.learning_experience
        metrics = manifest.normalized_metrics
        entry_price = exp.entry_price
        exit_price = exp.exit_price

        # ── Compute PnL ──
        direction = 1.0 if actual_side == "BUY" else -1.0
        actual_pnl: Optional[float] = None
        was_profitable: Optional[bool] = None
        if exit_price is not None and entry_price is not None:
            actual_pnl = (exit_price - entry_price) * direction
            was_profitable = actual_pnl > 0

        # ── Action alignment ──
        llm_action = capture.llm_action
        if llm_action in ("BUY", "SELL"):
            action_aligned = llm_action == actual_side
        else:
            action_aligned = False

        # ── Calibration ──
        confidence_vs_outcome = self._classify_calibration(
            llm_action=llm_action,
            llm_confidence=capture.llm_confidence,
            was_profitable=was_profitable,
            action_aligned=action_aligned,
        )

        # ── Evaluation notes ──
        notes: list[str] = []
        if was_profitable is False and metrics.mae_atr_multiple is not None and metrics.mae_atr_multiple < -2.0:
            notes.append(f"Large adverse movement: {metrics.mae_atr_multiple:.2f} ATR")
        if capture.evidence_source != "EXACT":
            notes.append(
                f"Decision used {capture.evidence_source} evidence "
                f"(tier {capture.evidence_tier})"
            )
        if actual_exit_reason and actual_exit_reason.startswith("STOP"):
            notes.append(f"Exit triggered by {actual_exit_reason}")
        if actual_pnl is not None and metrics.pnl_atr_multiple is not None:
            notes.append(
                f"PnL: {actual_pnl:+.4f} ({metrics.pnl_atr_multiple:+.2f} ATR)"
            )

        return DecisionEvaluation(
            position_id=exp.position_id,
            opportunity_id=exp.opportunity_id or capture.opportunity_id,
            candidate_id=capture.candidate_id,
            llm_action=capture.llm_action,
            llm_confidence=capture.llm_confidence,
            llm_rationale=capture.llm_rationale,
            llm_risk_assessment=capture.llm_risk_assessment,
            evidence_source=capture.evidence_source,
            evidence_tier=capture.evidence_tier,
            actual_side=actual_side,
            actual_quantity=actual_quantity,
            actual_entry_price=entry_price,
            actual_exit_price=exit_price,
            actual_pnl=actual_pnl,
            actual_pnl_atr=metrics.pnl_atr_multiple,
            actual_duration_minutes=metrics.holding_duration_minutes,
            actual_max_drawdown=exp.maximum_drawdown,
            actual_highest_profit=exp.highest_unrealized_profit,
            actual_exit_reason=actual_exit_reason,
            actual_integrity_score=manifest.validation_report.integrity_score,
            was_profitable=was_profitable,
            action_aligned=action_aligned,
            confidence_vs_outcome=confidence_vs_outcome,
            evaluation_notes=notes,
        )

    @staticmethod
    def _classify_calibration(
        llm_action: str,
        llm_confidence: float,
        was_profitable: Optional[bool],
        action_aligned: bool,
    ) -> str:
        if llm_action == "ABSTAIN":
            return "ABSTAINED"
        if llm_action == "HOLD":
            return "NO_TRADE"
        if not action_aligned:
            return "MISALIGNED"
        if was_profitable is None:
            return "UNKNOWN"

        if was_profitable:
            if llm_confidence >= 0.5:
                return "CORRECT_CALL"
            else:
                return "UNDERCERTAIN"
        else:
            if llm_confidence >= 0.5:
                return "OVERCERTAIN"
            else:
                return "APPROPRIATE_SKEPTIC"
