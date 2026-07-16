from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Optional

import structlog

import time

from src.models.learning.trade_experience import (
    ConfidenceScore,
    ConfigurationSnapshot,
    DuplicateResult,
    LearningExperience,
    LearningManifest,
    ManifestProvenance,
    NoiseAssessment,
    OpportunityIdentity,
    PersistenceVerification,
    PositionSnapshot,
)
from src.learning.extractor import ExperienceExtractor
from src.learning.validator import ExperienceValidator
from src.learning.normalizer import ExperienceNormalizer
from src.learning.feature_catalog import FeatureCatalog
from src.learning.config_catalog import ConfigurationCatalog
from src.learning.provenance import ProvenanceRegistry
from src.storage.learning.learning_corpus import (
    CandidateRejectionError,
    LearningCorpus,
    VerificationError,
)
from src.recommendations.models import LearningPolicy

logger = structlog.get_logger("learning_pipeline")


class MetadataResolver:
    """Pure assembly layer. Resolves feature IDs, builds config snapshot,
    constructs opportunity identity, resolves provenance version.
    No computation beyond deterministic identity generation."""

    def __init__(
        self,
        feature_catalog: FeatureCatalog,
        config_catalog: ConfigurationCatalog,
        provenance_registry: ProvenanceRegistry,
    ):
        self._feature_catalog = feature_catalog
        self._config_catalog = config_catalog
        self._provenance_registry = provenance_registry

    @property
    def feature_catalog_version(self) -> str:
        return self._feature_catalog.version

    @property
    def feature_catalog_hash(self) -> str:
        return self._feature_catalog.catalog_hash

    def resolve(
        self,
        snapshot: PositionSnapshot,
        runtime_config_values: Optional[dict[str, Any]] = None,
    ) -> tuple[ConfigurationSnapshot, OpportunityIdentity, ManifestProvenance, str]:
        config_snapshot = self._build_configuration_snapshot(runtime_config_values)
        opp_identity = self._build_opportunity_identity(snapshot)
        manifest_prov = self._build_manifest_provenance()
        prov_version = self._resolve_provenance_version()
        return config_snapshot, opp_identity, manifest_prov, prov_version

    def _build_configuration_snapshot(
        self, values: Optional[dict[str, Any]] = None,
    ) -> ConfigurationSnapshot:
        if values is None:
            return ConfigurationSnapshot()
        return self._config_catalog.build_runtime_snapshot(
            values, self._config_catalog,
            source="runtime", config_version=self._config_catalog.version,
        )

    def _build_opportunity_identity(self, snapshot: PositionSnapshot) -> OpportunityIdentity:
        opportunity_id = getattr(snapshot, "opportunity_id", "") or ""
        regime_parts = "|".join([
            snapshot.trend_regime or "UNKNOWN",
            snapshot.volatility_regime or "UNKNOWN",
            snapshot.correlation_regime or "UNKNOWN",
            snapshot.anchor_symbol,
            snapshot.timeframe,
        ])
        market_state_hash = hashlib.sha256(regime_parts.encode()).hexdigest()
        thesis_hash = hashlib.sha256(
            (snapshot.entry_thesis + snapshot.symbol + snapshot.timeframe).encode()
        ).hexdigest()

        return OpportunityIdentity(
            opportunity_id=opportunity_id,
            market_state_hash=market_state_hash,
            market_state_schema_version="1.0",
            scanner_version="",
            strategy_name="",
            strategy_version=snapshot.pipeline_version,
            discovered_at=getattr(snapshot, "frozen_at", datetime.utcnow()),
            anchor_symbol=snapshot.anchor_symbol,
            symbol=snapshot.symbol,
            timeframe=snapshot.timeframe,
            thesis_hash=thesis_hash,
        )

    def _build_manifest_provenance(self) -> ManifestProvenance:
        return ManifestProvenance(
            position_schema_version="1.0",
            event_schema_version="1.0",
            pipeline_version="1.0",
            git_commit="",
            build_id="",
            application_version="",
        )

    def _resolve_provenance_version(self) -> str:
        return self._provenance_registry.version


class LearningPipeline:
    """Orchestrates the deterministic learning pipeline.
    Stages are executed in order. Each stage produces an immutable artifact
    before passing it to the next stage."""

    def __init__(
        self,
        extractor: ExperienceExtractor,
        validator: ExperienceValidator,
        normalizer: ExperienceNormalizer,
        corpus: LearningCorpus,
        metadata_resolver: MetadataResolver,
        git_commit: str = "",
        build_id: str = "",
        application_version: str = "",
    ):
        self._extractor = extractor
        self._validator = validator
        self._normalizer = normalizer
        self._corpus = corpus
        self._metadata_resolver = metadata_resolver
        self._git_commit = git_commit
        self._build_id = build_id
        self._application_version = application_version

    async def process(
        self,
        snapshot: PositionSnapshot,
        runtime_config_values: Optional[dict[str, Any]] = None,
    ) -> LearningManifest:
        """Execute the full pipeline:
        MetadataAssembly → Extraction → Validation → Normalization → ManifestAssembly → Store."""

        # 1. Metadata Assembly
        config_snapshot, opp_identity, manifest_prov_base, prov_version = (
            self._metadata_resolver.resolve(snapshot, runtime_config_values)
        )

        # Stamp software identity into manifest_provenance
        base_dict = manifest_prov_base.model_dump()
        base_dict.pop("git_commit")
        base_dict.pop("build_id")
        base_dict.pop("application_version")
        manifest_prov = ManifestProvenance(
            **base_dict,
            git_commit=self._git_commit,
            build_id=self._build_id,
            application_version=self._application_version,
        )

        # 2. Extraction
        experience = self._extractor.extract(
            snapshot, opportunity_id=opp_identity.opportunity_id,
        )
        logger.info(
            "experience_extracted",
            experience_id=experience.experience_id,
            symbol=experience.symbol,
        )

        # 3. Validation (with metadata checks)
        metadata_for_validation = {
            "provenance_version": prov_version,
            "opportunity_id": opp_identity.opportunity_id,
            "feature_catalog_version": self._metadata_resolver.feature_catalog_version,
            "market_state_hash": opp_identity.market_state_hash,
        }
        report = self._validator.validate(experience, metadata=metadata_for_validation)
        logger.info(
            "experience_validated",
            experience_id=experience.experience_id,
            integrity_score=report.integrity_score,
            verified=len(report.verified_fields),
            missing=len(report.missing_fields),
        )

        # 4. Normalization
        metrics = self._normalizer.normalize(experience)

        # 5. Manifest Assembly
        manifest = LearningManifest(
            experience_id=experience.experience_id,
            position_id=experience.position_id,
            learning_experience=experience,
            validation_report=report,
            normalized_metrics=metrics,
            feature_catalog_version=self._metadata_resolver.feature_catalog_version,
            feature_catalog_hash=self._metadata_resolver.feature_catalog_hash,
            configuration_snapshot=config_snapshot,
            active_profile_id=getattr(snapshot, "active_profile_id", None),
            session_id=getattr(snapshot, "session_id", None),
            provenance_version=prov_version,
            opportunity_identity=opp_identity,
            manifest_provenance=manifest_prov,
        )

        # 6. Store
        self._corpus.save(manifest)

        logger.info(
            "experience_persisted",
            experience_id=manifest.experience_id,
            position_id=manifest.position_id,
            integrity_score=report.integrity_score,
            verified=len(report.verified_fields),
            missing=len(report.missing_fields),
            feature_catalog_hash=manifest.feature_catalog_hash[:12],
            provenance_version=manifest.provenance_version,
        )

        return manifest

    async def process_candidate(
        self,
        snapshot: PositionSnapshot,
        policy: LearningPolicy,
        runtime_config_values: Optional[dict[str, Any]] = None,
        decision_capture: Any = None,
        evaluation_engine: Any = None,
        evaluation_corpus: Any = None,
        config_store: Any = None,
        evidence_min_count_override: Optional[int] = None,
    ) -> dict:
        """Extended lifecycle:
        1. Manifest assembly (reuses process() without final save)
        2. Candidate Validation
        3. Duplicate Check
        4. Noise Assessment
        5. Confidence Assignment
        6. Evidence Threshold Check
        7. Persist with Verification
        8. Adaptive Recommendation Trigger
        """
        start = time.time()
        result: dict[str, Any] = {
            "position_id": snapshot.position_id,
            "candidate_created": False,
            "validation": None,
            "duplicate_result": None,
            "noise_score": None,
            "confidence": None,
            "evidence_count": 0,
            "stored": False,
            "verification": None,
            "adaptive_recommendation_triggered": False,
            "latency_ms": 0.0,
            "status": "pending",
        }

        # Stage 1: Manifest Assembly (reuse process internals without corpus.save)
        config_snapshot, opp_identity, manifest_prov_base, prov_version = (
            self._metadata_resolver.resolve(snapshot, runtime_config_values)
        )
        base_dict = manifest_prov_base.model_dump()
        base_dict.pop("git_commit", None)
        base_dict.pop("build_id", None)
        base_dict.pop("application_version", None)
        manifest_prov = ManifestProvenance(
            **base_dict,
            git_commit=self._git_commit,
            build_id=self._build_id,
            application_version=self._application_version,
        )

        experience = self._extractor.extract(
            snapshot, opportunity_id=opp_identity.opportunity_id,
        )
        metadata_for_validation = {
            "provenance_version": prov_version,
            "opportunity_id": opp_identity.opportunity_id,
            "feature_catalog_version": self._metadata_resolver.feature_catalog_version,
            "market_state_hash": opp_identity.market_state_hash,
        }
        report = self._validator.validate(experience, metadata=metadata_for_validation)
        metrics = self._normalizer.normalize(experience)

        manifest = LearningManifest(
            experience_id=experience.experience_id,
            position_id=experience.position_id,
            learning_experience=experience,
            validation_report=report,
            normalized_metrics=metrics,
            experience_type=experience.experience_type,
            feature_catalog_version=self._metadata_resolver.feature_catalog_version,
            feature_catalog_hash=self._metadata_resolver.feature_catalog_hash,
            configuration_snapshot=config_snapshot,
            active_profile_id=getattr(snapshot, "active_profile_id", None),
            session_id=getattr(snapshot, "session_id", None),
            provenance_version=prov_version,
            opportunity_identity=opp_identity,
            manifest_provenance=manifest_prov,
        )
        result["candidate_created"] = True

        # Stage 2: Candidate Validation
        if report.integrity_score < policy.validation_min_score:
            result["validation"] = report.integrity_score
            result["status"] = "rejected"
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            logger.info(
                "LEARNING_UPDATE",
                position_id=snapshot.position_id,
                candidate_created=True,
                validation=report.integrity_score,
                duplicate_result="skip",
                noise_score=0.0,
                confidence=0.0,
                evidence_count=0,
                stored=False,
                verification=False,
                adaptive_recommendation_triggered=False,
                latency_ms=result["latency_ms"],
                decision="rejected: low integrity",
            )
            raise CandidateRejectionError(
                f"Integrity score {report.integrity_score} below policy minimum {policy.validation_min_score}"
            )
        result["validation"] = report.integrity_score

        # Stage 3: Duplicate Check
        dup = self._check_duplicate(manifest, policy)
        result["duplicate_result"] = dup.model_dump(mode="json") if dup else None
        if dup and dup.is_duplicate:
            result["status"] = "duplicate"
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            if dup.matched_experience_id:
                new_count = self._corpus.increment_evidence(dup.matched_experience_id)
                result["evidence_count"] = new_count
            self._emit_learning_update(snapshot.position_id, result, decision="duplicate_merged")
            return result

        # Stage 4: Noise Assessment
        noise = self._assess_noise(manifest, policy, decision_capture)
        result["noise_score"] = noise.noise_score
        if noise.is_noise:
            result["status"] = "noise_rejected"
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            self._emit_learning_update(snapshot.position_id, result, decision=f"rejected: {noise.reason}")
            raise CandidateRejectionError(f"Noise rejection: {noise.reason}")

        # Stage 5: Confidence Assignment
        conf = self._assign_confidence(manifest, report, metrics, decision_capture)
        result["confidence"] = conf.score
        if conf.score < policy.confidence_min:
            result["status"] = "low_confidence"
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            self._emit_learning_update(snapshot.position_id, result, decision="rejected: low confidence")
            raise CandidateRejectionError(
                f"Confidence {conf.score:.3f} below policy minimum {policy.confidence_min}"
            )

        # Stage 6: Evidence Threshold Check
        evidence_count = self._count_evidence(manifest)
        result["evidence_count"] = evidence_count
        if evidence_count < policy.evidence_min_count:
            result["status"] = "pending"
            result["stored"] = False
            result["manifest"] = manifest
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            self._emit_learning_update(
                snapshot.position_id, result,
                decision=f"pending evidence ({evidence_count}/{policy.evidence_min_count})",
            )
            return result

        # Stage 7: Persist with Verification
        try:
            verification = self._corpus.save_with_verification(manifest, policy)
            result["verification"] = verification.model_dump(mode="json")
            result["stored"] = verification.verified
        except VerificationError as e:
            result["verification"] = {"verified": False, "error": str(e)}
            result["stored"] = False
            result["status"] = "verification_failed"
            result["latency_ms"] = round((time.time() - start) * 1000, 1)
            self._emit_learning_update(snapshot.position_id, result, decision="verification_failed")
            return result

        # Stage 8: Adaptive Recommendation Trigger
        if result["stored"] and config_store is not None:
            try:
                from src.recommendations.lifecycle import trigger_adaptive_recommendation
                triggered = trigger_adaptive_recommendation(
                    config_store=config_store,
                    manifest=manifest,
                    confidence=conf.score,
                    evidence_count=evidence_count,
                )
                result["adaptive_recommendation_triggered"] = triggered
            except Exception as e:
                logger.warning("Adaptive recommendation trigger failed", error=str(e))

        # Stage 8b: Decision Evaluation
        if result["stored"] and decision_capture is not None and evaluation_engine is not None and evaluation_corpus is not None:
            try:
                evaluation = evaluation_engine.evaluate(
                    manifest=manifest,
                    capture=decision_capture,
                    actual_side=snapshot.side,
                    actual_quantity=snapshot.quantity,
                    actual_exit_reason=snapshot.exit_reason,
                )
                if evaluation:
                    evaluation_corpus.save(evaluation)
                    result["evaluation"] = {
                        "was_profitable": evaluation.was_profitable,
                        "actual_pnl": evaluation.actual_pnl,
                        "confidence_vs_outcome": evaluation.confidence_vs_outcome,
                        "exit_reason": evaluation.actual_exit_reason,
                    }
            except Exception as e:
                logger.warning("Decision evaluation after candidate failed", error=str(e))

        result["status"] = "stored"
        result["manifest"] = manifest
        latency = time.time() - start
        result["latency_ms"] = round(latency * 1000, 1)

        self._emit_learning_update(snapshot.position_id, result, decision="stored")
        return result

    def _emit_learning_update(self, position_id: str, result: dict, decision: str) -> None:
        dup_result = result.get("duplicate_result")
        dup_label = "skip"
        if isinstance(dup_result, dict):
            dup_label = dup_result.get("is_duplicate", False)
        elif dup_result is not None:
            dup_label = getattr(dup_result, "is_duplicate", "skip")

        verification = result.get("verification") or {}
        verified = verification.get("verified", False) if isinstance(verification, dict) else False

        logger.info(
            "LEARNING_UPDATE",
            position_id=position_id,
            candidate_created=result.get("candidate_created", False),
            validation=result.get("validation"),
            duplicate_result=dup_label,
            noise_score=result.get("noise_score"),
            confidence=result.get("confidence"),
            evidence_count=result.get("evidence_count", 0),
            stored=result.get("stored", False),
            verification=verified,
            adaptive_recommendation_triggered=result.get("adaptive_recommendation_triggered", False),
            latency_ms=result.get("latency_ms", 0.0),
            decision=decision,
        )

    def _check_duplicate(self, manifest: LearningManifest, policy: LearningPolicy) -> Optional[DuplicateResult]:
        oi = manifest.opportunity_identity
        if oi is None:
            return None
        le = manifest.learning_experience
        market_hash = oi.market_state_hash
        symbol = le.symbol
        timeframe = le.timeframe
        side = le.side or getattr(manifest, "side", "")

        matching = self._corpus.get_corpus_view(limit=1000)
        for record in matching:
            if record.symbol != symbol or record.timeframe != timeframe:
                continue
            # Skip self-position matches — interim experiences for the same
            # position should not block the final experience from being stored.
            if record.position_id == manifest.position_id:
                continue
            record_side = getattr(record, "side", "") or ""
            if record_side and side and record_side != side:
                continue
            record_market_hash = getattr(record, "market_state_hash", "")
            if not record_market_hash:
                continue
            if record_market_hash == market_hash:
                return DuplicateResult(
                    is_duplicate=True,
                    matched_experience_id=getattr(record, "experience_id", ""),
                    match_score=1.0,
                    matched_fields=["market_state_hash", "symbol", "timeframe", "side"],
                    action="merge",
                )
        return DuplicateResult(is_duplicate=False, match_score=0.0)

    def _assess_noise(
        self, manifest: LearningManifest, policy: LearningPolicy, decision_capture: Any = None,
    ) -> NoiseAssessment:
        factors: list[str] = []
        report = manifest.validation_report

        if report.integrity_score < 60:
            factors.append("low_integrity_score")

        if decision_capture is None:
            factors.append("no_decision_capture")

        if not factors:
            return NoiseAssessment(is_noise=False, noise_score=0.0, reason="")

        score = min(1.0, len(factors) * 0.25)
        threshold = policy.noise_max_score
        return NoiseAssessment(
            is_noise=score > threshold,
            noise_score=score,
            noise_factors=factors,
            reason="; ".join(factors),
        )

    def _assign_confidence(
        self, manifest: LearningManifest, report: Any, metrics: Any, decision_capture: Any = None,
    ) -> ConfidenceScore:
        eval_quality = report.integrity_score / 100.0 if report else 0.5
        exec_integrity = 0.5
        if manifest.learning_experience.slippage_bps is not None:
            max_expected = 20.0
            exec_integrity = max(0.0, 1.0 - (manifest.learning_experience.slippage_bps / max_expected))
        evidence_strength = 0.5
        if manifest.learning_experience.episode_count > 0:
            evidence_strength = min(1.0, manifest.learning_experience.episode_count / 10.0)
        data_completeness = 0.5
        if report:
            total = len(report.verified_fields) + len(report.missing_fields)
            data_completeness = len(report.verified_fields) / max(total, 1)
        consistency = 0.7

        score = (
            eval_quality * 0.25
            + exec_integrity * 0.20
            + evidence_strength * 0.25
            + data_completeness * 0.15
            + consistency * 0.15
        )
        return ConfidenceScore(
            score=round(min(1.0, max(0.0, score)), 4),
            evaluation_quality=round(eval_quality, 4),
            execution_integrity=round(exec_integrity, 4),
            evidence_strength=round(evidence_strength, 4),
            data_completeness=round(data_completeness, 4),
            consistency=round(consistency, 4),
        )

    def _count_evidence(self, manifest: LearningManifest) -> int:
        market_hash = ""
        if manifest.opportunity_identity:
            market_hash = manifest.opportunity_identity.market_state_hash
        episode = max(1, manifest.learning_experience.episode_count)
        if market_hash:
            accumulated = self._corpus.count_similar_evidence(market_hash)
            return accumulated + 1
        return episode
