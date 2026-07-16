from __future__ import annotations

import structlog

from src.evaluation.models import DecisionCapture
from src.core.models import SystemEvent
from src.evaluation.storage import EvaluationCorpus

logger = structlog.get_logger("decision_capture_store")


class DecisionCaptureStore:
    """Subscribes to CANDIDATE_EVALUATED events and persists decision data
    keyed by opportunity_id for later retrieval at evaluation time.
    Persisted via EvaluationCorpus — survives restart, crash, power loss."""

    def __init__(self, corpus: EvaluationCorpus):
        self._corpus = corpus

    async def _on_candidate_evaluated(self, event: SystemEvent) -> None:
        payload = event.payload
        candidate = payload.get("candidate", {})
        llm_decision = payload.get("llm_decision", {})

        capture = DecisionCapture(
            opportunity_id=candidate.get("opportunity_id", ""),
            candidate_id=payload.get("candidate_id", ""),
            symbol=candidate.get("symbol", ""),
            llm_action=llm_decision.get("action", "ABSTAIN"),
            llm_confidence=llm_decision.get("confidence", 0.0),
            llm_rationale=llm_decision.get("rationale", ""),
            llm_risk_assessment=llm_decision.get("risk_assessment", ""),
            evidence_source=payload.get("evidence_source", "COLD_START"),
            evidence_tier=payload.get("evidence_tier", 4),
        )

        key = capture.opportunity_id
        if not key:
            logger.warning(
                "DecisionCapture skipped — no opportunity_id",
                candidate_id=capture.candidate_id,
            )
            return

        self._corpus.save_capture(capture)

    def get(self, opportunity_id: str) -> DecisionCapture | None:
        return self._corpus.get_capture(opportunity_id)

    def __len__(self) -> int:
        return len(self._corpus._conn.execute(
            "SELECT COUNT(*) FROM decision_captures"
        ).fetchone()[0])