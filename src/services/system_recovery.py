from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import structlog

from src.models.learning.trade_experience import (
    LearningManifest,
    PositionSnapshot,
)
from src.learning.pipeline import LearningPipeline
from src.storage.learning.learning_corpus import LearningCorpus, VerificationError
from src.evaluation.storage import EvaluationCorpus
from src.evaluation.store import DecisionCaptureStore
from src.evaluation.engine import DecisionEvaluationEngine
from src.services.portfolio_manager import PortfolioManager
from src.recommendations.store import ConfigurationStore
from src.recommendations.models import LearningPolicy

logger = structlog.get_logger("system_recovery")


class SystemRecoveryService:
    """Recovers learning artifacts for closed positions that lack them.

    This handles the case where a position was closed during a prior session
    (or via reconciliation) but no LearningManifest or Evaluation was ever
    persisted. It runs idempotently — skipping positions that already have artifacts.
    """

    def __init__(
        self,
        portfolio_mgr: PortfolioManager,
        learning_pipeline: LearningPipeline,
        learning_corpus: LearningCorpus,
        evaluation_corpus: EvaluationCorpus,
        decision_capture_store: DecisionCaptureStore,
        evaluation_engine: DecisionEvaluationEngine,
    ):
        self._portfolio_mgr = portfolio_mgr
        self._learning_pipeline = learning_pipeline
        self._learning_corpus = learning_corpus
        self._evaluation_corpus = evaluation_corpus
        self._decision_capture_store = decision_capture_store
        self._evaluation_engine = evaluation_engine

    async def recover_orphaned_positions(
        self,
        runtime_config_values: Optional[dict[str, Any]] = None,
    ) -> int:
        if runtime_config_values is None:
            runtime_config_values = self._default_runtime_config()

        closed_states = {"CLOSED", "ARCHIVED"}
        missing_positions = self._portfolio_mgr.get_terminal_positions()
        recovered_count = 0

        for pos in missing_positions:
            existing_manifest = self._learning_corpus.find_by_position_id(pos.position_id)
            if existing_manifest:
                continue
            existing_eval = self._evaluation_corpus.get_by_position_id(pos.position_id)
            if existing_eval is not None:
                continue
            try:
                snapshot = PositionSnapshot.from_position(pos)
                manifest = await self._learning_pipeline.process(
                    snapshot, runtime_config_values=runtime_config_values,
                )

                opp_id = (
                    manifest.opportunity_identity.opportunity_id
                    if manifest.opportunity_identity else ""
                )
                capture = self._decision_capture_store.get(opp_id) if opp_id else None
                if capture:
                    evaluation = self._evaluation_engine.evaluate(
                        manifest=manifest,
                        capture=capture,
                        actual_side=pos.side,
                        actual_quantity=pos.quantity,
                        actual_exit_reason=pos.exit_reason or "ORPHANED_RECONCILIATION",
                    )
                    if evaluation:
                        self._evaluation_corpus.save(evaluation)

                recovered_count += 1
            except Exception as e:
                logger.warning(
                    "Learning recovery failed for position",
                    position_id=pos.position_id,
                    symbol=pos.symbol,
                    error=str(e),
                )

        if recovered_count > 0:
            logger.info(
                "Startup learning recovery complete",
                recovered=recovered_count,
            )

        return recovered_count

    async def full_system_recovery(
        self,
        corpus: LearningCorpus,
        config_store: ConfigurationStore,
        policy: Optional[LearningPolicy] = None,
        runtime_config_values: Optional[dict[str, Any]] = None,
    ) -> dict:
        """Comprehensive recovery: candidates, writes, verifications, maintenance.
        Returns a structured result for SYSTEM_RECOVERY logging."""
        result: dict[str, Any] = {
            "recovered_candidates": 0,
            "recovered_writes": 0,
            "recovered_verifications": 0,
            "integrity_result": "PASS",
            "remaining_issues": [],
            "recovered_orphaned": 0,
        }

        if policy is None:
            policy = LearningPolicy(name="RecoveryDefault", tier="balanced")

        # 1. Recover orphaned positions (existing logic)
        try:
            orphaned = await self.recover_orphaned_positions(runtime_config_values)
            result["recovered_orphaned"] = orphaned
            if orphaned:
                logger.info("Recovered orphaned positions during full recovery", count=orphaned)
        except Exception as e:
            result["remaining_issues"].append(f"orphan_recovery: {e}")
            logger.warning("Full recovery: orphan recovery failed", error=str(e))

        # 2. Recover pending candidates — check if any can be promoted
        try:
            pending = corpus.get_pending_candidates()
            for cand in pending:
                try:
                    cid = cand.get("candidate_id", "")
                    evidence = cand.get("evidence_count", 1)
                    if evidence >= policy.evidence_min_count:
                        corpus.update_candidate_status(cid, "promoting")
                        manifest_raw = cand.get("manifest_json", {})
                        if isinstance(manifest_raw, str):
                            manifest_raw = json.loads(manifest_raw)
                        manifest = LearningManifest(**manifest_raw)
                        try:
                            corpus.save_with_verification(manifest, policy)
                            corpus.update_candidate_status(cid, "promoted")
                            result["recovered_candidates"] += 1
                        except VerificationError:
                            corpus.update_candidate_status(cid, "failed")
                            result["remaining_issues"].append(f"candidate_verification_failed:{cid[:12]}")
                except Exception as e:
                    result["remaining_issues"].append(f"candidate_recovery_error:{e}")
                    continue
        except Exception as e:
            result["remaining_issues"].append(f"pending_candidates: {e}")

        # 3. Complete interrupted writes
        try:
            stuck = corpus.get_pending_candidates()
            for cand in stuck:
                if cand.get("status") in ("promoting", "writing"):
                    cid = cand.get("candidate_id", "")
                    manifest_raw = cand.get("manifest_json", {})
                    if isinstance(manifest_raw, str):
                        manifest_raw = json.loads(manifest_raw)
                    manifest = LearningManifest(**manifest_raw)
                    try:
                        ver = corpus.verify_persistence(manifest.experience_id)
                        if ver.verified:
                            corpus.update_candidate_status(cid, "verified")
                            result["recovered_writes"] += 1
                        else:
                            corpus.update_candidate_status(cid, "failed")
                            result["remaining_issues"].append(f"write_verification_failed:{cid[:12]}")
                    except Exception:
                        corpus.update_candidate_status(cid, "failed")
                        continue
        except Exception as e:
            result["remaining_issues"].append(f"interrupted_writes: {e}")

        # 4. Resume interrupted maintenance
        try:
            health = corpus.get_memory_health()
            now = datetime.utcnow()
            if health.last_maintenance:
                last_maint = datetime.fromisoformat(health.last_maintenance.replace("Z", "+00:00"))
                hours_since = (now - last_maint).total_seconds() / 3600
                if hours_since > policy.maintenance_interval_hours * 2:
                    await corpus.run_maintenance(policy)
                    result["remaining_issues"].append("maintenance_resumed_after_interruption")
            else:
                await corpus.run_maintenance(policy)
        except Exception as e:
            result["remaining_issues"].append(f"maintenance: {e}")

        # 5. Final memory integrity verification
        try:
            health = corpus.get_memory_health()
            result["integrity_result"] = "PASS" if health.integrity_state == "ok" else "DEGRADED"
            if health.integrity_state != "ok":
                result["remaining_issues"].append("memory_integrity_degraded")
        except Exception as e:
            result["integrity_result"] = "FAIL"
            result["remaining_issues"].append(f"integrity_check: {e}")

        logger.info(
            "SYSTEM_RECOVERY",
            recovered_orphaned=result["recovered_orphaned"],
            recovered_candidates=result["recovered_candidates"],
            recovered_writes=result["recovered_writes"],
            integrity_result=result["integrity_result"],
            remaining_issues=result["remaining_issues"],
            result="PASS" if not result["remaining_issues"] else "PARTIAL",
        )
        return result

    @staticmethod
    def _default_runtime_config() -> dict[str, Any]:
        return {
            "execution.leverage": 5,
            "execution.sizing_mode": "fixed_usdt",
            "execution.sizing_value": 5.0,
            "risk.max_concurrent_positions": 6,
            "risk.max_live_exposure_usdt": 25000,
            "risk.min_llm_confidence": 0.3,
            "execution.stop_loss_pct": 0.98,
            "execution.take_profit_pct": 1.04,
            "execution.trailing_stop_atr_mult": 2.0,
            "execution.spread_bps": 2.0,
            "execution.fee_bps": 4.0,
            "execution.slippage_bps": 3.0,
            "scanner.min_correlation": 0.6,
            "scanner.max_p_value": 0.05,
        }
