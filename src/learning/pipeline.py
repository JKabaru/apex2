from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional

import structlog

from src.models.learning.trade_experience import (
    ConfigurationSnapshot,
    LearningExperience,
    LearningManifest,
    ManifestProvenance,
    OpportunityIdentity,
    PositionSnapshot,
)
from src.learning.extractor import ExperienceExtractor
from src.learning.validator import ExperienceValidator
from src.learning.normalizer import ExperienceNormalizer
from src.learning.feature_catalog import FeatureCatalog
from src.learning.config_catalog import ConfigurationCatalog
from src.learning.provenance import ProvenanceRegistry
from src.storage.learning.learning_corpus import LearningCorpus

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
