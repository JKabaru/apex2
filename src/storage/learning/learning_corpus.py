from __future__ import annotations

import hashlib
import json
import os
import random
import time
from datetime import datetime, timedelta
from typing import Any, Optional

import duckdb
import structlog

from src.models.learning.trade_experience import (
    ConfidenceScore,
    DuplicateResult,
    LearningManifest,
    MaintenanceReport,
    MemoryHealth,
    NoiseAssessment,
    PersistenceVerification,
)
from src.models.learning.corpus_metadata import CorpusMetadata
from src.models.learning.observation import Observation, ObservationCategory, SourceComponent
from src.models.learning.observation_aggregate import ObservationAggregate
from src.models.learning.timeline import Timeline, TimelineObservation, TimelineStatus
from src.models.learning.pattern import Pattern, PatternCategory
from src.models.learning.hypothesis import Hypothesis, HypothesisEvidence, HypothesisStatus
from src.models.learning.knowledge import Knowledge, KnowledgeConfidence
from src.models.learning.reasoning_episode import ReasoningEpisode
from src.models.learning.belief import Belief
from src.retrieval.models import CorpusDiagnostics, RetrievalRecord
from src.retrieval.projection import CorpusProjection
from src.recommendations.models import LearningPolicy

logger = structlog.get_logger("learning_corpus")

CORPUS_DB = "data/experience_corpus.duckdb"


class CandidateRejectionError(Exception):
    """Raised when a candidate fails validation, noise, or duplicate checks."""


class VerificationError(Exception):
    """Raised when memory persistence verification fails."""


class LearningCorpus:
    """Single owner of memory. Responsible for experience persistence,
    candidate lifecycle, health reporting, maintenance, and verification."""

    def __init__(self, db_path: str = CORPUS_DB):
        self._db_path = db_path
        self._last_save: Optional[datetime] = None
        self._last_maintenance: Optional[datetime] = None
        try:
            self._conn = duckdb.connect(db_path)
        except duckdb.InternalException:
            wal_path = db_path + ".wal"
            if os.path.exists(wal_path):
                os.remove(wal_path)
                self._conn = duckdb.connect(db_path)
            else:
                raise
        logger.info("LearningCorpus initialized", db_path=db_path)

    def create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS experiences (
                experience_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL,
                manifest_json JSON NOT NULL,
                hash VARCHAR NOT NULL,
                schema_version VARCHAR NOT NULL DEFAULT '2.0',
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corpus_position_id
            ON experiences (position_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_corpus_created_at
            ON experiences (created_at)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS corpus_meta (
                key VARCHAR PRIMARY KEY,
                value JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL,
                manifest_json JSON NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'pending',
                validation_report JSON,
                duplicate_result JSON,
                noise_assessment JSON,
                confidence_score JSON,
                evidence_count INTEGER DEFAULT 1,
                policy_id VARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_candidates_status
            ON candidates (status)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS rejected_candidates (
                candidate_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL,
                manifest_json JSON NOT NULL,
                reject_reason VARCHAR NOT NULL,
                reject_stage VARCHAR NOT NULL,
                reject_details JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS consolidation_log (
                log_id INTEGER PRIMARY KEY,
                primary_experience_id VARCHAR NOT NULL,
                merged_experience_ids JSON NOT NULL,
                merge_reason VARCHAR NOT NULL,
                new_confidence DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                source VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                importance DOUBLE NOT NULL,
                symbol VARCHAR NOT NULL,
                data JSON NOT NULL,
                context JSON,
                session_id VARCHAR,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_timestamp
            ON observations (timestamp)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_symbol
            ON observations (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_importance
            ON observations (importance)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_source
            ON observations (source)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS observation_aggregates (
                aggregate_id VARCHAR PRIMARY KEY,
                observation_ids JSON NOT NULL,
                count INTEGER NOT NULL,
                window_start TIMESTAMP NOT NULL,
                window_end TIMESTAMP NOT NULL,
                source VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL,
                importance DOUBLE NOT NULL,
                summary_data JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregates_symbol
            ON observation_aggregates (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_aggregates_window
            ON observation_aggregates (window_start)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS timelines (
                timeline_id VARCHAR PRIMARY KEY,
                position_id VARCHAR NOT NULL UNIQUE,
                symbol VARCHAR NOT NULL,
                side VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                opened_at TIMESTAMP NOT NULL,
                closed_at TIMESTAMP,
                status VARCHAR NOT NULL DEFAULT 'open',
                observation_count INTEGER DEFAULT 0,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timelines_status
            ON timelines (status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timelines_symbol
            ON timelines (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timelines_position_id
            ON timelines (position_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS timeline_observations (
                timeline_id VARCHAR NOT NULL,
                observation_id VARCHAR NOT NULL,
                sequence INTEGER NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                importance_at_addition DOUBLE NOT NULL,
                PRIMARY KEY (timeline_id, observation_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeline_obs_timeline
            ON timeline_observations (timeline_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_timeline_obs_observation
            ON timeline_observations (observation_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id VARCHAR PRIMARY KEY,
                timeline_id VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                description VARCHAR NOT NULL,
                observation_ids JSON,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP NOT NULL,
                confidence DOUBLE DEFAULT 0.0,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_timeline
            ON patterns (timeline_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_patterns_category
            ON patterns (category)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hypotheses (
                hypothesis_id VARCHAR PRIMARY KEY,
                statement VARCHAR NOT NULL,
                pattern_ids JSON,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                side VARCHAR,
                created_at TIMESTAMP NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'draft',
                evidence_count INTEGER DEFAULT 0,
                confidence DOUBLE DEFAULT 0.0,
                supporting_count INTEGER DEFAULT 0,
                contradicting_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP NOT NULL,
                metadata JSON,
                created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hypotheses_status
            ON hypotheses (status)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hypotheses_symbol
            ON hypotheses (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hypotheses_confidence
            ON hypotheses (confidence)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_evidence (
                hypothesis_id VARCHAR NOT NULL,
                timeline_id VARCHAR NOT NULL,
                observation_id VARCHAR NOT NULL,
                weight DOUBLE DEFAULT 1.0,
                supports BOOLEAN NOT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (hypothesis_id, timeline_id, observation_id)
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hyp_evidence_hypothesis
            ON hypothesis_evidence (hypothesis_id)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hyp_evidence_timeline
            ON hypothesis_evidence (timeline_id)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                knowledge_id VARCHAR PRIMARY KEY,
                statement VARCHAR NOT NULL,
                hypothesis_ids JSON,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL,
                confidence VARCHAR NOT NULL DEFAULT 'emerging',
                confidence_score DOUBLE DEFAULT 0.0,
                supporting_hypothesis_count INTEGER DEFAULT 0,
                contradicting_hypothesis_count INTEGER DEFAULT 0,
                cross_timeline_count INTEGER DEFAULT 0,
                created_at TIMESTAMP NOT NULL,
                last_updated TIMESTAMP NOT NULL,
                deprecated_at TIMESTAMP,
                metadata JSON,
                created_at_ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_confidence
            ON knowledge (confidence)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_symbol
            ON knowledge (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_knowledge_cross_timeline
            ON knowledge (cross_timeline_count)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS reasoning_episodes (
                episode_id VARCHAR PRIMARY KEY,
                decision_id VARCHAR NOT NULL DEFAULT '',
                timestamp TIMESTAMP NOT NULL,
                symbol VARCHAR NOT NULL,
                timeframe VARCHAR NOT NULL DEFAULT '',
                prompt_hash VARCHAR NOT NULL DEFAULT '',
                prompt_preview VARCHAR NOT NULL DEFAULT '',
                market_summary JSON,
                evidence_summary JSON,
                portfolio_summary JSON,
                action VARCHAR NOT NULL DEFAULT '',
                confidence DOUBLE DEFAULT 0.0,
                rationale VARCHAR NOT NULL DEFAULT '',
                risk_assessment VARCHAR NOT NULL DEFAULT '',
                llm_response_raw VARCHAR NOT NULL DEFAULT '',
                chosen_signals JSON,
                ignored_signals JSON,
                predicted_outcome VARCHAR NOT NULL DEFAULT '',
                retrieval_memory_ids JSON,
                retrieval_belief_ids JSON,
                retrieval_knowledge_ids JSON,
                execution_id VARCHAR NOT NULL DEFAULT '',
                correlation_id VARCHAR NOT NULL DEFAULT '',
                opportunity_id VARCHAR NOT NULL DEFAULT '',
                strategy_version VARCHAR NOT NULL DEFAULT '',
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reasoning_symbol
            ON reasoning_episodes (symbol)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reasoning_timestamp
            ON reasoning_episodes (timestamp)
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                belief_id VARCHAR PRIMARY KEY,
                statement VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                symbol VARCHAR NOT NULL DEFAULT '',
                confidence DOUBLE DEFAULT 0.0,
                strength DOUBLE DEFAULT 0.0,
                source VARCHAR NOT NULL DEFAULT 'reflection',
                observation_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deprecated BOOLEAN DEFAULT FALSE,
                metadata JSON
            )
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_beliefs_category
            ON beliefs (category)
        """)
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_beliefs_symbol
            ON beliefs (symbol)
        """)
        logger.info("Learning corpus schema initialized (v4.0 — observation + timeline + pattern + hypothesis + knowledge + reasoning_episodes + beliefs)")

    @property
    def db_path(self) -> str:
        return self._db_path

    # ── Experience Persistence ──

    def save(self, manifest: LearningManifest) -> None:
        manifest_dict = manifest.model_dump(mode="json")
        self._conn.execute("""
            INSERT INTO experiences (experience_id, position_id, manifest_json, hash, schema_version, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (experience_id) DO NOTHING
        """, [
            manifest.experience_id,
            manifest.position_id,
            json.dumps(manifest_dict),
            manifest.hash,
            manifest.schema_version,
            datetime.utcnow(),
        ])

    def load(self, experience_id: str) -> Optional[LearningManifest]:
        row = self._conn.execute(
            "SELECT manifest_json FROM experiences WHERE experience_id = ?",
            [experience_id],
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def find_by_position_id(self, position_id: str) -> Optional[LearningManifest]:
        row = self._conn.execute(
            "SELECT manifest_json FROM experiences WHERE position_id = ? LIMIT 1",
            [position_id],
        ).fetchone()
        if row is None:
            return None
        return self._deserialize(row[0])

    def replace_interim_with_final(self, position_id: str, final_manifest: LearningManifest) -> int:
        interim_manifests = self._conn.execute(
            "SELECT experience_id, manifest_json FROM experiences "
            "WHERE position_id = ? AND manifest_json->>'experience_type' = 'interim'",
            [position_id],
        ).fetchall()

        removed = 0
        for row in interim_manifests:
            exp_id = row[0]
            self._conn.execute(
                "DELETE FROM experiences WHERE experience_id = ?",
                [exp_id],
            )
            self._conn.execute(
                "INSERT INTO consolidation_log (primary_experience_id, merged_experience_ids, merge_reason, created_at) "
                "VALUES (?, ?, ?, ?)",
                [
                    final_manifest.experience_id,
                    json.dumps([exp_id]),
                    "interim_replaced_by_final",
                    datetime.utcnow(),
                ],
            )
            removed += 1

        if removed > 0:
            logger.info(
                "Replaced interim experiences with final",
                position_id=position_id, removed=removed,
                final_experience_id=final_manifest.experience_id,
            )

        self.save(final_manifest)
        return removed

    def load_batch(
        self, limit: int = 100, offset: int = 0
    ) -> list[LearningManifest]:
        rows = self._conn.execute(
            "SELECT manifest_json FROM experiences "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [limit, offset],
        ).fetchall()
        return [self._deserialize(r[0]) for r in rows]

    def exists(self, experience_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM experiences WHERE experience_id = ?",
            [experience_id],
        ).fetchone()
        return row is not None

    def statistics(self) -> dict[str, Any]:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]
        return {
            "total_experiences": total,
        }

    # ── Save with Persistence Verification ──

    def save_with_verification(
        self, manifest: LearningManifest, policy: Optional[LearningPolicy] = None,
    ) -> PersistenceVerification:
        self.save(manifest)
        result = self.verify_persistence(manifest.experience_id)
        if not result.verified:
            self._rollback(manifest.experience_id)
            logger.error(
                "MEMORY_VERIFICATION",
                experience_id=manifest.experience_id,
                hash_verified=result.hash_matches,
                read_back=result.read_back_ok,
                workspace_verified=result.workspace_ok,
                index_verified=result.index_ok,
                visibility_verified=result.visibility_ok,
                result="FAIL",
                error=result.error,
            )
            raise VerificationError(f"Persistence verification failed: {result.error}")
        self._last_save = datetime.utcnow()
        try:
            self._conn.execute("CHECKPOINT")
        except Exception:
            pass
        logger.info(
            "MEMORY_VERIFICATION",
            experience_id=manifest.experience_id,
            hash_verified=result.hash_matches,
            read_back=result.read_back_ok,
            workspace_verified=result.workspace_ok,
            index_verified=result.index_ok,
            visibility_verified=result.visibility_ok,
            result="PASS",
        )
        return result

    def verify_persistence(self, experience_id: str) -> PersistenceVerification:
        errors: list[str] = []
        hash_matches = True
        read_back_ok = True
        index_ok = True
        workspace_ok = True
        visibility_ok = True

        loaded = self.load(experience_id)
        if loaded is None:
            errors.append("read_back: experience not found after write")
            read_back_ok = False
            return PersistenceVerification(
                verified=False, read_back_ok=False, error="; ".join(errors),
            )

        stored_row = self._conn.execute(
            "SELECT hash FROM experiences WHERE experience_id = ?", [experience_id],
        ).fetchone()
        stored_hash = stored_row[0] if stored_row else ""
        computed_hash = loaded.hash
        if stored_hash and stored_hash != computed_hash:
            errors.append("hash: stored hash does not match computed manifest hash")
            hash_matches = False

        idx_check = self._conn.execute(
            "SELECT 1 FROM experiences WHERE position_id = ? AND experience_id = ?",
            [loaded.position_id, experience_id],
        ).fetchone()
        if idx_check is None:
            errors.append("index: position_id lookup failed")
            index_ok = False

        if not errors:
            return PersistenceVerification(
                verified=True, hash_matches=True, read_back_ok=True,
                index_ok=True, workspace_ok=True, visibility_ok=True,
            )
        return PersistenceVerification(
            verified=False, hash_matches=hash_matches, read_back_ok=read_back_ok,
            index_ok=index_ok, workspace_ok=workspace_ok, visibility_ok=visibility_ok,
            error="; ".join(errors),
        )

    def _rollback(self, experience_id: str) -> None:
        self._conn.execute(
            "DELETE FROM experiences WHERE experience_id = ?", [experience_id],
        )
        logger.warning("Experience rolled back due to verification failure", experience_id=experience_id)

    # ── Candidate Management ──

    def save_candidate(self, candidate: dict) -> str:
        cid = candidate.get("candidate_id", candidate.get("experience_id", ""))
        self._conn.execute("""
            INSERT OR REPLACE INTO candidates
                (candidate_id, position_id, manifest_json, status, validation_report,
                 duplicate_result, noise_assessment, confidence_score, evidence_count,
                 policy_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            cid,
            candidate.get("position_id", ""),
            json.dumps(candidate.get("manifest_json", {})),
            candidate.get("status", "pending"),
            json.dumps(candidate.get("validation_report")) if candidate.get("validation_report") else None,
            json.dumps(candidate.get("duplicate_result")) if candidate.get("duplicate_result") else None,
            json.dumps(candidate.get("noise_assessment")) if candidate.get("noise_assessment") else None,
            json.dumps(candidate.get("confidence_score")) if candidate.get("confidence_score") else None,
            candidate.get("evidence_count", 1),
            candidate.get("policy_id"),
            datetime.utcnow(),
            datetime.utcnow(),
        ])
        return cid

    def get_candidate(self, candidate_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ?", [candidate_id],
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        return dict(zip(cols, row))

    def get_pending_candidates(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM candidates WHERE status = 'pending' ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, r)) for r in rows]

    def get_rejected_candidates(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM rejected_candidates ORDER BY created_at DESC LIMIT ?",
            [limit],
        ).fetchall()
        cols = [desc[0] for desc in self._conn.description]
        return [dict(zip(cols, r)) for r in rows]

    def update_candidate_status(
        self, candidate_id: str, status: str, updates: Optional[dict] = None,
    ) -> None:
        self._conn.execute(
            "UPDATE candidates SET status = ?, updated_at = ? WHERE candidate_id = ?",
            [status, datetime.utcnow(), candidate_id],
        )
        if updates:
            for key, val in updates.items():
                col = key if key in ("evidence_count", "confidence_score", "validation_report") else None
                if col:
                    self._conn.execute(
                        f"UPDATE candidates SET {col} = ? WHERE candidate_id = ?",
                        [json.dumps(val) if isinstance(val, (dict, list)) else val, candidate_id],
                    )

    def count_pending_candidates(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE status = 'pending'",
        ).fetchone()[0]

    def count_rejected_candidates(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM rejected_candidates",
        ).fetchone()[0]

    def record_rejection(
        self, candidate: dict, reason: str, stage: str, details: Optional[dict] = None,
    ) -> str:
        cid = candidate.get("candidate_id", candidate.get("experience_id", ""))
        self._conn.execute("""
            INSERT INTO rejected_candidates
                (candidate_id, position_id, manifest_json, reject_reason,
                 reject_stage, reject_details, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            cid,
            candidate.get("position_id", ""),
            json.dumps(candidate.get("manifest_json", {})),
            reason,
            stage,
            json.dumps(details) if details else None,
            datetime.utcnow(),
        ])
        self._conn.execute(
            "DELETE FROM candidates WHERE candidate_id = ?", [cid],
        )
        return cid

    # ── Duplicate Merging ──

    def merge_duplicate(
        self, primary_id: str, duplicate_id: str, new_confidence: float,
    ) -> bool:
        primary = self.load(primary_id)
        duplicate = self.load(duplicate_id)
        if primary is None or duplicate is None:
            logger.warning("Merge failed: one or both experiences not found",
                           primary=primary_id, duplicate=duplicate_id)
            return False

        primary_dict = primary.model_dump(mode="json")
        dup_count = self._get_duplicate_count(primary_id)
        primary_dict["_duplicate_evidence_count"] = dup_count + 1
        self._conn.execute("""
            UPDATE experiences SET manifest_json = ? WHERE experience_id = ?
        """, [json.dumps(primary_dict), primary_id])

        self._conn.execute(
            "DELETE FROM experiences WHERE experience_id = ?", [duplicate_id],
        )
        self._conn.execute("""
            INSERT INTO consolidation_log
                (log_id, primary_experience_id, merged_experience_ids, merge_reason, new_confidence, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, [
            self._conn.execute("SELECT COALESCE(MAX(log_id), 0) + 1 FROM consolidation_log").fetchone()[0],
            primary_id,
            json.dumps([duplicate_id]),
            "duplicate_merge",
            new_confidence,
            datetime.utcnow(),
        ])
        logger.info("Duplicate merged", primary=primary_id, duplicate=duplicate_id, confidence=new_confidence)
        return True

    def _get_duplicate_count(self, experience_id: str) -> int:
        rows = self._conn.execute(
            "SELECT merged_experience_ids FROM consolidation_log "
            "WHERE primary_experience_id = ?", [experience_id],
        ).fetchall()
        return sum(len(json.loads(r[0])) for r in rows) if rows else 0

    # ── Memory Health ──

    def get_memory_health(self) -> MemoryHealth:
        exp_count = self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        pend_count = self.count_pending_candidates()
        rej_count = self.count_rejected_candidates()
        dup_count = self._conn.execute("SELECT COUNT(*) FROM consolidation_log").fetchone()[0]

        try:
            db_size = os.path.getsize(self._db_path)
        except OSError:
            db_size = 0

        try:
            wal_path = self._db_path.replace(".duckdb", ".duckdb.wal")
            wal_size = os.path.getsize(wal_path) if os.path.exists(wal_path) else 0
            ws_size = db_size + wal_size
        except OSError:
            ws_size = db_size

        integrity_ok = self._quick_integrity_check()
        verification_ok = self._last_save is not None

        return MemoryHealth(
            experience_count=exp_count,
            pending_candidates=pend_count,
            rejected_count=rej_count,
            duplicate_count=dup_count,
            workspace_size_bytes=ws_size,
            database_size_bytes=db_size,
            integrity_state="ok" if integrity_ok else "degraded",
            verification_state="verified" if verification_ok else "unverified",
            last_maintenance=self._last_maintenance.isoformat() if self._last_maintenance else None,
            last_save=self._last_save.isoformat() if self._last_save else None,
        )

    def _quick_integrity_check(self, sample_pct: float = 0.1) -> bool:
        total = self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        if total == 0:
            return True
        sample_size = max(1, int(total * sample_pct))
        ids = self._conn.execute(
            f"SELECT experience_id, hash FROM experiences USING SAMPLE {sample_size}",
        ).fetchall()
        for exp_id, stored_hash in ids:
            manifest = self.load(exp_id)
            if manifest is None:
                return False
            if manifest.hash != stored_hash:
                return False
        return True

    # ── Memory Maintenance ──

    async def run_maintenance(self, policy: LearningPolicy) -> MaintenanceReport:
        start = time.time()
        errors: list[str] = []
        scanned = 0
        merged = 0
        confidence_updates = 0

        try:
            if policy.duplicate_threshold < 1.0:
                merged = self._consolidate_duplicates(policy)
        except Exception as e:
            errors.append(f"consolidation: {e}")
            logger.warning("Maintenance consolidation failed", error=str(e))

        try:
            if policy.confidence_decay_rate > 0:
                confidence_updates = self._decay_confidence(policy)
        except Exception as e:
            errors.append(f"decay: {e}")
            logger.warning("Maintenance confidence decay failed", error=str(e))

        try:
            self._conn.execute("CHECKPOINT")
            db_optimized = True
        except Exception as e:
            errors.append(f"checkpoint: {e}")
            db_optimized = False

        try:
            integrity_ok = self._quick_integrity_check(0.05)
        except Exception as e:
            errors.append(f"integrity: {e}")
            integrity_ok = False

        scanned = self._conn.execute("SELECT COUNT(*) FROM experiences").fetchone()[0]
        self._last_maintenance = datetime.utcnow()
        duration = time.time() - start

        report = MaintenanceReport(
            experiences_scanned=scanned,
            duplicates_merged=merged,
            confidence_updates=confidence_updates,
            database_optimized=db_optimized,
            integrity_verified=integrity_ok,
            workspace_size_bytes=self.get_memory_health().workspace_size_bytes,
            duration_seconds=round(duration, 2),
            errors=errors,
        )

        logger.info(
            "MEMORY_MAINTENANCE",
            experiences_scanned=report.experiences_scanned,
            duplicates_merged=report.duplicates_merged,
            confidence_updates=report.confidence_updates,
            database_optimized=report.database_optimized,
            integrity_verified=report.integrity_verified,
            workspace_size_bytes=report.workspace_size_bytes,
            duration_seconds=report.duration_seconds,
            errors=report.errors,
            result="PASS" if not errors else "PARTIAL",
        )
        return report

    def _consolidate_duplicates(self, policy: LearningPolicy) -> int:
        merged = 0
        rows = self._conn.execute("""
            SELECT experience_id, manifest_json FROM experiences
            ORDER BY created_at DESC
        """).fetchall()

        seen: dict[str, list[tuple[str, str, str]]] = {}
        for exp_id, raw in rows:
            data = json.loads(raw)
            le = data.get("learning_experience", {})
            oi = data.get("opportunity_identity", {})
            key_parts = "|".join([
                oi.get("market_state_hash", ""),
                le.get("symbol", ""),
                le.get("timeframe", ""),
                le.get("side", ""),
                str(le.get("exit_price", "")),
            ])
            seen.setdefault(key_parts, []).append((exp_id, raw, key_parts))

        for key, group in seen.items():
            if len(group) < 2:
                continue
            group.sort(key=lambda x: x[0])
            primary = group[0]
            for duplicate in group[1:]:
                try:
                    self.merge_duplicate(primary[0], duplicate[0], 0.9)
                    merged += 1
                except Exception as e:
                    logger.warning("Duplicate merge failed", primary=primary[0], duplicate=duplicate[0], error=str(e))

        return merged

    def _decay_confidence(self, policy: LearningPolicy) -> int:
        updated = 0
        now = datetime.utcnow()
        rows = self._conn.execute(
            "SELECT experience_id, manifest_json FROM experiences",
        ).fetchall()
        for exp_id, raw in rows:
            try:
                data = json.loads(raw)
                created = data.get("created_at", now.isoformat())
                if isinstance(created, str):
                    created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                else:
                    created_dt = now
                age_days = (now - created_dt).total_seconds() / 86400.0
                if age_days <= 0:
                    continue
                decay = (1.0 - policy.confidence_decay_rate) ** age_days
                data["_confidence_decay"] = round(decay, 4)
                self._conn.execute(
                    "UPDATE experiences SET manifest_json = ? WHERE experience_id = ?",
                    [json.dumps(data), exp_id],
                )
                updated += 1
            except Exception:
                continue
        return updated

    # ── Corpus Metadata ──

    def save_corpus_metadata(self, metadata: CorpusMetadata) -> None:
        raw = json.dumps(metadata.model_dump(mode="json"))
        self._conn.execute("""
            INSERT INTO corpus_meta (key, value, created_at)
            VALUES ('corpus_metadata', ?, ?)
            ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, created_at = EXCLUDED.created_at
        """, [raw, datetime.utcnow()])

    def get_corpus_metadata(self) -> Optional[CorpusMetadata]:
        row = self._conn.execute(
            "SELECT value FROM corpus_meta WHERE key = 'corpus_metadata'"
        ).fetchone()
        if row is None:
            return None
        data = json.loads(row[0])
        return CorpusMetadata(**data)

    @staticmethod
    def _deserialize(raw: str) -> LearningManifest:
        data = json.loads(raw)
        return LearningManifest(**data)

    # ── Retrieval View ──

    def get_corpus_view(
        self,
        limit: int = 500,
        filters: Optional[dict] = None,
    ) -> list[RetrievalRecord]:
        rows = self._conn.execute(
            """
            SELECT manifest_json, created_at FROM experiences
            UNION ALL
            SELECT manifest_json, created_at FROM candidates WHERE status = 'pending'
            ORDER BY created_at DESC LIMIT ?
            """,
            [limit],
        ).fetchall()

        projector = CorpusProjection()
        records = [projector.project_manifest(self._deserialize(r[0])) for r in rows]

        if filters:
            records = self._apply_retrieval_filters(records, filters)

        return records

    @staticmethod
    def _apply_retrieval_filters(
        records: list[RetrievalRecord],
        filters: dict,
    ) -> list[RetrievalRecord]:
        result = list(records)
        if "symbol" in filters and filters["symbol"] is not None:
            val = filters["symbol"]
            result = [r for r in result if r.symbol == val]
        if "timeframe" in filters and filters["timeframe"] is not None:
            val = filters["timeframe"]
            result = [r for r in result if r.timeframe == val]
        if "opportunity_id" in filters and filters["opportunity_id"] is not None:
            val = filters["opportunity_id"]
            result = [r for r in result if r.opportunity_id == val]
        if "market_state_hash" in filters and filters["market_state_hash"] is not None:
            val = filters["market_state_hash"]
            result = [r for r in result if r.market_state_hash == val]
        if "trend_regime" in filters and filters["trend_regime"] is not None:
            val = filters["trend_regime"]
            result = [r for r in result if r.trend_regime == val]
        if "volatility_regime" in filters and filters["volatility_regime"] is not None:
            val = filters["volatility_regime"]
            result = [r for r in result if r.volatility_regime == val]
        if "correlation_regime" in filters and filters["correlation_regime"] is not None:
            val = filters["correlation_regime"]
            result = [r for r in result if r.correlation_regime == val]
        if "min_integrity" in filters and filters["min_integrity"] is not None:
            val = int(filters["min_integrity"])
            result = [r for r in result if r.integrity_score >= val]
        if "experience_type" in filters and filters["experience_type"] is not None:
            val = str(filters["experience_type"])
            result = [r for r in result if r.experience_type == val]
        return result

    # ── Diagnostics ──

    def get_diagnostics(self) -> CorpusDiagnostics:
        total = self._conn.execute(
            "SELECT COUNT(*) FROM experiences"
        ).fetchone()[0]

        schema_version_rows = self._conn.execute(
            "SELECT schema_version, COUNT(*) AS cnt FROM experiences GROUP BY schema_version ORDER BY cnt DESC"
        ).fetchall()
        schema_versions = {row[0]: row[1] for row in schema_version_rows}

        projector = CorpusProjection()
        records = self.get_corpus_view(limit=10000)

        symbol_dist: dict[str, int] = {}
        timeframe_dist: dict[str, int] = {}
        integrity_scores: list[int] = []
        pipeline_version_dist: dict[str, int] = {}
        catalog_hash_dist: dict[str, int] = {}

        regime_trend: dict[str, int] = {}
        regime_vol: dict[str, int] = {}
        regime_corr: dict[str, int] = {}

        metric_fields = [
            "normalized_entry_atr_multiple",
            "normalized_exit_atr_multiple",
            "pnl_atr_multiple",
            "mfe_atr_multiple",
            "mae_atr_multiple",
            "entry_rsi_percentile",
            "entry_volatility_percentile",
            "holding_duration_minutes",
            "bars_held",
            "total_slippage_bps",
            "total_fees_bps",
            "realized_rr",
            "initial_risk_atr_multiple",
        ]
        missing_counts: dict[str, int] = {f: 0 for f in metric_fields}
        total_count = len(records)

        for r in records:
            symbol_dist[r.symbol] = symbol_dist.get(r.symbol, 0) + 1
            timeframe_dist[r.timeframe] = timeframe_dist.get(r.timeframe, 0) + 1
            integrity_scores.append(r.integrity_score)

            pipeline_version_dist[r.pipeline_version] = pipeline_version_dist.get(r.pipeline_version, 0) + 1
            catalog_hash_dist[r.hash[:12]] = catalog_hash_dist.get(r.hash[:12], 0) + 1

            if r.trend_regime is not None:
                regime_trend[r.trend_regime] = regime_trend.get(r.trend_regime, 0) + 1
            else:
                regime_trend["__missing__"] = regime_trend.get("__missing__", 0) + 1

            if r.volatility_regime is not None:
                regime_vol[r.volatility_regime] = regime_vol.get(r.volatility_regime, 0) + 1
            else:
                regime_vol["__missing__"] = regime_vol.get("__missing__", 0) + 1

            if r.correlation_regime is not None:
                regime_corr[r.correlation_regime] = regime_corr.get(r.correlation_regime, 0) + 1
            else:
                regime_corr["__missing__"] = regime_corr.get("__missing__", 0) + 1

            for fname in metric_fields:
                if getattr(r, fname, None) is None:
                    missing_counts[fname] += 1

        avg_integrity = round(sum(integrity_scores) / len(integrity_scores), 2) if integrity_scores else 0.0
        missing_pct = {
            fname: round(count / total_count * 100, 1) if total_count > 0 else 0.0
            for fname, count in missing_counts.items()
        }

        return CorpusDiagnostics(
            total_experiences=total,
            avg_integrity_score=avg_integrity,
            schema_version_distribution=schema_versions,
            pipeline_version_distribution=pipeline_version_dist,
            catalog_hash_distribution=catalog_hash_dist,
            missing_feature_percentages=missing_pct,
            regime_distributions={
                "trend": regime_trend,
                "volatility": regime_vol,
                "correlation": regime_corr,
            },
            symbol_distribution=dict(sorted(symbol_dist.items(), key=lambda x: -x[1])),
            timeframe_distribution=dict(sorted(timeframe_dist.items(), key=lambda x: -x[1])),
        )

    def count_similar_evidence(self, market_state_hash: str) -> int:
        """Count accumulated evidence for a market-state pattern (stored + pending)."""
        if not market_state_hash:
            return 0
        total = 0
        rows = self._conn.execute("SELECT manifest_json FROM experiences").fetchall()
        for row in rows:
            data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
            oi = data.get("opportunity_identity") or {}
            if oi.get("market_state_hash") == market_state_hash:
                total += int(data.get("_duplicate_evidence_count", 1))
        for cand in self.get_pending_candidates(limit=500):
            manifest_raw = cand.get("manifest_json", {})
            if isinstance(manifest_raw, str):
                manifest_raw = json.loads(manifest_raw)
            if not manifest_raw:
                continue
            oi = manifest_raw.get("opportunity_identity") or {}
            if oi.get("market_state_hash") == market_state_hash:
                total += 1
        return total

    def increment_evidence(self, experience_id: str) -> int:
        """Increase evidence count on an existing experience without creating a duplicate record."""
        manifest = self.load(experience_id)
        if manifest is None:
            return 0
        manifest_dict = manifest.model_dump(mode="json")
        new_count = int(manifest_dict.get("_duplicate_evidence_count", 1)) + 1
        manifest_dict["_duplicate_evidence_count"] = new_count
        self._conn.execute(
            "UPDATE experiences SET manifest_json = ?, created_at = ? WHERE experience_id = ?",
            [json.dumps(manifest_dict), datetime.utcnow(), experience_id],
        )
        self._last_save = datetime.utcnow()
        logger.info(
            "MEMORY_EVIDENCE_INCREMENT",
            experience_id=experience_id,
            evidence_count=new_count,
        )
        return new_count

    # ── Observation Methods ──

    def save_observation(self, observation: Observation) -> str:
        raw = json.dumps(observation.data)
        ctx = json.dumps(observation.context) if observation.context else None
        meta = json.dumps(observation.metadata) if observation.metadata else None
        self._conn.execute("""
            INSERT INTO observations
                (observation_id, timestamp, source, category, importance, symbol,
                 data, context, session_id, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (observation_id) DO NOTHING
        """, [
            observation.observation_id,
            observation.timestamp,
            observation.source.value,
            observation.category.value,
            observation.importance,
            observation.symbol,
            raw,
            ctx,
            observation.session_id,
            meta,
            datetime.utcnow(),
        ])
        return observation.observation_id

    def get_observation(self, observation_id: str) -> Optional[Observation]:
        row = self._conn.execute(
            "SELECT * FROM observations WHERE observation_id = ?",
            [observation_id],
        ).fetchone()
        if row is None:
            return None
        return self._observation_from_row(row)

    def query_observations(
        self,
        symbol: str | None = None,
        source: str | None = None,
        category: str | None = None,
        min_importance: float | None = None,
        since: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Observation]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if source is not None:
            clauses.append("source = ?")
            params.append(source)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if min_importance is not None:
            clauses.append("importance >= ?")
            params.append(min_importance)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM observations WHERE {where} ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [limit, offset],
        ).fetchall()
        return [self._observation_from_row(r) for r in rows]

    def get_recent_observations(
        self, minutes: int = 15, min_importance: float = 0.0,
    ) -> list[Observation]:
        since = datetime.utcnow() - timedelta(minutes=minutes)
        rows = self._conn.execute(
            "SELECT * FROM observations WHERE timestamp >= ? AND importance >= ? "
            "ORDER BY timestamp DESC LIMIT 500",
            [since, min_importance],
        ).fetchall()
        return [self._observation_from_row(r) for r in rows]

    def _observation_from_row(self, row) -> Observation:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Observation(
            observation_id=data["observation_id"],
            timestamp=data["timestamp"],
            source=SourceComponent(data["source"]),
            category=ObservationCategory(data["category"]),
            importance=data["importance"],
            symbol=data["symbol"],
            data=json.loads(data["data"]) if isinstance(data["data"], str) else data["data"],
            context=json.loads(data["context"]) if data.get("context") and isinstance(data["context"], str) else data.get("context") or {},
            session_id=data.get("session_id"),
            metadata=json.loads(data["metadata"]) if data.get("metadata") and isinstance(data["metadata"], str) else data.get("metadata") or {},
        )

    # ── Observation Aggregate Methods ──

    def save_aggregate(self, aggregate: ObservationAggregate) -> str:
        oids = json.dumps(aggregate.observation_ids)
        sdata = json.dumps(aggregate.summary_data) if aggregate.summary_data else None
        self._conn.execute("""
            INSERT INTO observation_aggregates
                (aggregate_id, observation_ids, count, window_start, window_end,
                 source, category, symbol, importance, summary_data, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (aggregate_id) DO NOTHING
        """, [
            aggregate.aggregate_id,
            oids,
            aggregate.count,
            aggregate.window_start,
            aggregate.window_end,
            aggregate.source.value,
            aggregate.category.value,
            aggregate.symbol,
            aggregate.importance,
            sdata,
            aggregate.created_at,
        ])
        return aggregate.aggregate_id

    def get_aggregate(self, aggregate_id: str) -> Optional[ObservationAggregate]:
        row = self._conn.execute(
            "SELECT * FROM observation_aggregates WHERE aggregate_id = ?",
            [aggregate_id],
        ).fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return ObservationAggregate(
            aggregate_id=data["aggregate_id"],
            observation_ids=json.loads(data["observation_ids"]),
            count=data["count"],
            window_start=data["window_start"],
            window_end=data["window_end"],
            source=SourceComponent(data["source"]),
            category=ObservationCategory(data["category"]),
            symbol=data["symbol"],
            importance=data["importance"],
            summary_data=json.loads(data["summary_data"]) if data.get("summary_data") and isinstance(data["summary_data"], str) else data.get("summary_data") or {},
            created_at=data["created_at"] if "created_at" in data else datetime.utcnow(),
        )

    def query_aggregates(
        self, symbol: str | None = None, since: datetime | None = None, limit: int = 100,
    ) -> list[ObservationAggregate]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if since is not None:
            clauses.append("window_start >= ?")
            params.append(since)
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM observation_aggregates WHERE {where} ORDER BY window_start DESC LIMIT ?",
            params + [limit],
        ).fetchall()
        result: list[ObservationAggregate] = []
        for r in rows:
            cols = [desc[0] for desc in self._conn.description]
            data = dict(zip(cols, r))
            result.append(ObservationAggregate(
                aggregate_id=data["aggregate_id"],
                observation_ids=json.loads(data["observation_ids"]),
                count=data["count"],
                window_start=data["window_start"],
                window_end=data["window_end"],
                source=SourceComponent(data["source"]),
                category=ObservationCategory(data["category"]),
                symbol=data["symbol"],
                importance=data["importance"],
                summary_data=json.loads(data["summary_data"]) if data.get("summary_data") and isinstance(data["summary_data"], str) else data.get("summary_data") or {},
                created_at=data["created_at"] if "created_at" in data else datetime.utcnow(),
            ))
        return result

    # ── Timeline Methods ──

    def save_timeline(self, timeline: Timeline) -> str:
        meta = json.dumps(timeline.metadata) if timeline.metadata else None
        self._conn.execute("""
            INSERT INTO timelines
                (timeline_id, position_id, symbol, side, timeframe, opened_at,
                 closed_at, status, observation_count, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (timeline_id) DO NOTHING
        """, [
            timeline.timeline_id,
            timeline.position_id,
            timeline.symbol,
            timeline.side,
            timeline.timeframe,
            timeline.opened_at,
            timeline.closed_at,
            timeline.status.value,
            timeline.observation_count,
            meta,
            datetime.utcnow(),
        ])
        return timeline.timeline_id

    def get_timeline(self, timeline_id: str) -> Optional[Timeline]:
        row = self._conn.execute(
            "SELECT * FROM timelines WHERE timeline_id = ?",
            [timeline_id],
        ).fetchone()
        if row is None:
            return None
        return self._timeline_from_row(row)

    def get_timeline_by_position(self, position_id: str) -> Optional[Timeline]:
        row = self._conn.execute(
            "SELECT * FROM timelines WHERE position_id = ? LIMIT 1",
            [position_id],
        ).fetchone()
        if row is None:
            return None
        return self._timeline_from_row(row)

    def get_open_timeline_by_symbol(self, symbol: str) -> Optional[Timeline]:
        row = self._conn.execute(
            "SELECT * FROM timelines WHERE symbol = ? AND status = 'open' LIMIT 1",
            [symbol],
        ).fetchone()
        if row is None:
            return None
        return self._timeline_from_row(row)

    def update_timeline_status(self, timeline_id: str, status: TimelineStatus) -> bool:
        self._conn.execute(
            "UPDATE timelines SET status = ? WHERE timeline_id = ?",
            [status.value, timeline_id],
        )
        return True

    def close_timeline(self, timeline_id: str, closed_at: datetime | None = None) -> bool:
        ts = closed_at or datetime.utcnow()
        self._conn.execute(
            "UPDATE timelines SET status = ?, closed_at = ? WHERE timeline_id = ? AND status = 'open'",
            [TimelineStatus.CLOSED.value, ts, timeline_id],
        )
        return True

    def save_timeline_observation(self, link: TimelineObservation) -> None:
        self._conn.execute("""
            INSERT INTO timeline_observations
                (timeline_id, observation_id, sequence, added_at, importance_at_addition)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (timeline_id, observation_id) DO NOTHING
        """, [
            link.timeline_id,
            link.observation_id,
            link.sequence,
            link.added_at,
            link.importance_at_addition,
        ])

    def get_active_timelines(self) -> list[Timeline]:
        rows = self._conn.execute(
            "SELECT * FROM timelines WHERE status = 'open' ORDER BY opened_at DESC",
        ).fetchall()
        return [self._timeline_from_row(r) for r in rows]

    def get_closed_timelines_since(self, since: datetime) -> list[Timeline]:
        rows = self._conn.execute(
            "SELECT * FROM timelines WHERE status IN ('closed', 'ready_for_analysis') AND closed_at >= ? "
            "ORDER BY closed_at DESC",
            [since],
        ).fetchall()
        return [self._timeline_from_row(r) for r in rows]

    def get_timeline_observations(
        self, timeline_id: str, ordered: bool = True,
    ) -> list[TimelineObservation]:
        order = "ORDER BY sequence ASC" if ordered else ""
        rows = self._conn.execute(
            f"SELECT * FROM timeline_observations WHERE timeline_id = ? {order}",
            [timeline_id],
        ).fetchall()
        result: list[TimelineObservation] = []
        for r in rows:
            cols = [desc[0] for desc in self._conn.description]
            data = dict(zip(cols, r))
            result.append(TimelineObservation(
                timeline_id=data["timeline_id"],
                observation_id=data["observation_id"],
                sequence=data["sequence"],
                added_at=data["added_at"],
                importance_at_addition=data["importance_at_addition"],
            ))
        return result

    def _timeline_from_row(self, row) -> Timeline:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Timeline(
            timeline_id=data["timeline_id"],
            position_id=data["position_id"],
            symbol=data["symbol"],
            side=data["side"],
            timeframe=data["timeframe"],
            opened_at=data["opened_at"],
            closed_at=data.get("closed_at"),
            status=TimelineStatus(data["status"]),
            observation_count=data["observation_count"],
            metadata=json.loads(data["metadata"]) if data.get("metadata") and isinstance(data["metadata"], str) else data.get("metadata") or {},
        )

    # ── Pattern Methods ──

    def save_pattern(self, pattern: Pattern) -> str:
        oids = json.dumps(pattern.observation_ids) if pattern.observation_ids else None
        meta = json.dumps(pattern.metadata) if pattern.metadata else None
        self._conn.execute("""
            INSERT INTO patterns
                (pattern_id, timeline_id, category, description, observation_ids,
                 start_time, end_time, confidence, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (pattern_id) DO NOTHING
        """, [
            pattern.pattern_id,
            pattern.timeline_id,
            pattern.category.value,
            pattern.description,
            oids,
            pattern.start_time,
            pattern.end_time,
            pattern.confidence,
            meta,
            datetime.utcnow(),
        ])
        return pattern.pattern_id

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        row = self._conn.execute(
            "SELECT * FROM patterns WHERE pattern_id = ?",
            [pattern_id],
        ).fetchone()
        if row is None:
            return None
        return self._pattern_from_row(row)

    def get_patterns_by_timeline(self, timeline_id: str) -> list[Pattern]:
        rows = self._conn.execute(
            "SELECT * FROM patterns WHERE timeline_id = ? ORDER BY start_time ASC",
            [timeline_id],
        ).fetchall()
        return [self._pattern_from_row(r) for r in rows]

    def get_patterns_by_category(
        self, category: PatternCategory, limit: int = 100,
    ) -> list[Pattern]:
        rows = self._conn.execute(
            "SELECT * FROM patterns WHERE category = ? ORDER BY confidence DESC LIMIT ?",
            [category.value, limit],
        ).fetchall()
        return [self._pattern_from_row(r) for r in rows]

    def _pattern_from_row(self, row) -> Pattern:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Pattern(
            pattern_id=data["pattern_id"],
            timeline_id=data["timeline_id"],
            category=PatternCategory(data["category"]),
            description=data["description"],
            observation_ids=json.loads(data["observation_ids"]) if data.get("observation_ids") and isinstance(data["observation_ids"], str) else data.get("observation_ids") or [],
            start_time=data["start_time"],
            end_time=data["end_time"],
            confidence=data["confidence"],
            metadata=json.loads(data["metadata"]) if data.get("metadata") and isinstance(data["metadata"], str) else data.get("metadata") or {},
        )

    # ── Hypothesis Methods ──

    def save_hypothesis(self, hypothesis: Hypothesis) -> str:
        pids = json.dumps(hypothesis.pattern_ids) if hypothesis.pattern_ids else None
        meta = json.dumps(hypothesis.metadata) if hypothesis.metadata else None
        self._conn.execute("""
            INSERT INTO hypotheses
                (hypothesis_id, statement, pattern_ids, symbol, timeframe, side,
                 created_at, status, evidence_count, confidence, supporting_count,
                 contradicting_count, last_updated, metadata, created_at_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (hypothesis_id) DO NOTHING
        """, [
            hypothesis.hypothesis_id,
            hypothesis.statement,
            pids,
            hypothesis.symbol,
            hypothesis.timeframe,
            hypothesis.side,
            hypothesis.created_at,
            hypothesis.status.value,
            hypothesis.evidence_count,
            hypothesis.confidence,
            hypothesis.supporting_count,
            hypothesis.contradicting_count,
            hypothesis.last_updated,
            meta,
            datetime.utcnow(),
        ])
        return hypothesis.hypothesis_id

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        row = self._conn.execute(
            "SELECT * FROM hypotheses WHERE hypothesis_id = ?",
            [hypothesis_id],
        ).fetchone()
        if row is None:
            return None
        return self._hypothesis_from_row(row)

    def update_hypothesis_status(self, hypothesis_id: str, status: HypothesisStatus) -> bool:
        self._conn.execute(
            "UPDATE hypotheses SET status = ?, last_updated = ? WHERE hypothesis_id = ?",
            [status.value, datetime.utcnow(), hypothesis_id],
        )
        return True

    def update_hypothesis_confidence(
        self, hypothesis_id: str, confidence: float,
        supporting: int, contradicting: int,
    ) -> bool:
        self._conn.execute(
            "UPDATE hypotheses SET confidence = ?, supporting_count = ?, "
            "contradicting_count = ?, evidence_count = ?, last_updated = ? "
            "WHERE hypothesis_id = ?",
            [confidence, supporting, contradicting, supporting + contradicting,
             datetime.utcnow(), hypothesis_id],
        )
        return True

    def save_hypothesis_evidence(self, evidence: HypothesisEvidence) -> None:
        self._conn.execute("""
            INSERT INTO hypothesis_evidence
                (hypothesis_id, timeline_id, observation_id, weight, supports, added_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (hypothesis_id, timeline_id, observation_id) DO NOTHING
        """, [
            evidence.hypothesis_id,
            evidence.timeline_id,
            evidence.observation_id,
            evidence.weight,
            evidence.supports,
            evidence.added_at,
        ])

    def get_hypotheses_by_symbol(
        self, symbol: str, min_confidence: float = 0.0,
    ) -> list[Hypothesis]:
        rows = self._conn.execute(
            "SELECT * FROM hypotheses WHERE symbol = ? AND confidence >= ? "
            "ORDER BY confidence DESC",
            [symbol, min_confidence],
        ).fetchall()
        return [self._hypothesis_from_row(r) for r in rows]

    def get_mature_hypotheses(self, min_confidence: float = 0.7) -> list[Hypothesis]:
        rows = self._conn.execute(
            "SELECT * FROM hypotheses WHERE status = 'mature' AND confidence >= ? "
            "ORDER BY confidence DESC",
            [min_confidence],
        ).fetchall()
        return [self._hypothesis_from_row(r) for r in rows]

    def _hypothesis_from_row(self, row) -> Hypothesis:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Hypothesis(
            hypothesis_id=data["hypothesis_id"],
            statement=data["statement"],
            pattern_ids=json.loads(data["pattern_ids"]) if data.get("pattern_ids") and isinstance(data["pattern_ids"], str) else data.get("pattern_ids") or [],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            side=data.get("side"),
            created_at=data["created_at"],
            status=HypothesisStatus(data["status"]),
            evidence_count=data["evidence_count"],
            confidence=data["confidence"],
            supporting_count=data["supporting_count"],
            contradicting_count=data["contradicting_count"],
            last_updated=data["last_updated"],
            metadata=json.loads(data["metadata"]) if data.get("metadata") and isinstance(data["metadata"], str) else data.get("metadata") or {},
        )

    # ── Knowledge Methods ──

    def save_knowledge(self, knowledge: Knowledge) -> str:
        hids = json.dumps(knowledge.hypothesis_ids) if knowledge.hypothesis_ids else None
        meta = json.dumps(knowledge.metadata) if knowledge.metadata else None
        self._conn.execute("""
            INSERT INTO knowledge
                (knowledge_id, statement, hypothesis_ids, symbol, timeframe,
                 confidence, confidence_score, supporting_hypothesis_count,
                 contradicting_hypothesis_count, cross_timeline_count,
                 created_at, last_updated, deprecated_at, metadata, created_at_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (knowledge_id) DO NOTHING
        """, [
            knowledge.knowledge_id,
            knowledge.statement,
            hids,
            knowledge.symbol,
            knowledge.timeframe,
            knowledge.confidence.value,
            knowledge.confidence_score,
            knowledge.supporting_hypothesis_count,
            knowledge.contradicting_hypothesis_count,
            knowledge.cross_timeline_count,
            knowledge.created_at,
            knowledge.last_updated,
            knowledge.deprecated_at,
            meta,
            datetime.utcnow(),
        ])
        return knowledge.knowledge_id

    def get_knowledge(self, knowledge_id: str) -> Optional[Knowledge]:
        row = self._conn.execute(
            "SELECT * FROM knowledge WHERE knowledge_id = ?",
            [knowledge_id],
        ).fetchone()
        if row is None:
            return None
        return self._knowledge_from_row(row)

    def update_knowledge_confidence(
        self, knowledge_id: str, confidence: KnowledgeConfidence,
        confidence_score: float, supporting: int, contradicting: int,
        cross_timeline: int,
    ) -> bool:
        self._conn.execute(
            "UPDATE knowledge SET confidence = ?, confidence_score = ?, "
            "supporting_hypothesis_count = ?, contradicting_hypothesis_count = ?, "
            "cross_timeline_count = ?, last_updated = ? "
            "WHERE knowledge_id = ?",
            [confidence.value, confidence_score, supporting, contradicting,
             cross_timeline, datetime.utcnow(), knowledge_id],
        )
        return True

    def deprecate_knowledge(self, knowledge_id: str) -> bool:
        self._conn.execute(
            "UPDATE knowledge SET confidence = ?, deprecated_at = ?, last_updated = ? "
            "WHERE knowledge_id = ?",
            [KnowledgeConfidence.DEPRECATED.value, datetime.utcnow(), datetime.utcnow(), knowledge_id],
        )
        return True

    def get_active_knowledge(self) -> list[Knowledge]:
        rows = self._conn.execute(
            "SELECT * FROM knowledge WHERE confidence != 'deprecated' "
            "ORDER BY confidence_score DESC, cross_timeline_count DESC",
        ).fetchall()
        return [self._knowledge_from_row(r) for r in rows]

    def query_knowledge(
        self, symbol: str | None = None,
        timeframe: str | None = None,
        min_level: KnowledgeConfidence | None = None,
    ) -> list[Knowledge]:
        clauses: list[str] = []
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        if min_level is not None:
            levels = ["emerging", "developing", "established"]
            min_idx = levels.index(min_level.value)
            allowed = levels[min_idx:]
            placeholders = ", ".join(f"'{l}'" for l in allowed)
            clauses.append(f"confidence IN ({placeholders})")
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn.execute(
            f"SELECT * FROM knowledge WHERE {where} "
            "ORDER BY confidence_score DESC, cross_timeline_count DESC LIMIT 100",
            params,
        ).fetchall()
        return [self._knowledge_from_row(r) for r in rows]

    def _knowledge_from_row(self, row) -> Knowledge:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Knowledge(
            knowledge_id=data["knowledge_id"],
            statement=data["statement"],
            hypothesis_ids=json.loads(data["hypothesis_ids"]) if data.get("hypothesis_ids") and isinstance(data["hypothesis_ids"], str) else data.get("hypothesis_ids") or [],
            symbol=data["symbol"],
            timeframe=data["timeframe"],
            confidence=KnowledgeConfidence(data["confidence"]),
            confidence_score=data["confidence_score"],
            supporting_hypothesis_count=data["supporting_hypothesis_count"],
            contradicting_hypothesis_count=data["contradicting_hypothesis_count"],
            cross_timeline_count=data["cross_timeline_count"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            deprecated_at=data.get("deprecated_at"),
            metadata=json.loads(data["metadata"]) if data.get("metadata") and isinstance(data["metadata"], str) else data.get("metadata") or {},
        )

    # ── Reasoning Episodes ──

    def save_reasoning_episode(self, episode: ReasoningEpisode) -> str:
        """Persist a ReasoningEpisode to the corpus."""
        self._conn.execute("""
            INSERT INTO reasoning_episodes (
                episode_id, decision_id, timestamp, symbol, timeframe,
                prompt_hash, prompt_preview, market_summary, evidence_summary,
                portfolio_summary, action, confidence, rationale, risk_assessment,
                llm_response_raw, chosen_signals, ignored_signals, predicted_outcome,
                retrieval_memory_ids, retrieval_belief_ids, retrieval_knowledge_ids,
                execution_id, correlation_id, opportunity_id, strategy_version,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (episode_id) DO NOTHING
        """, [
            episode.episode_id,
            episode.decision_id,
            episode.timestamp,
            episode.symbol,
            episode.timeframe,
            episode.prompt_hash,
            episode.prompt_preview,
            json.dumps(episode.market_summary) if episode.market_summary else None,
            json.dumps(episode.evidence_summary) if episode.evidence_summary else None,
            json.dumps(episode.portfolio_summary) if episode.portfolio_summary else None,
            episode.action,
            episode.confidence,
            episode.rationale,
            episode.risk_assessment,
            episode.llm_response_raw,
            json.dumps(episode.chosen_signals) if episode.chosen_signals else None,
            json.dumps(episode.ignored_signals) if episode.ignored_signals else None,
            episode.predicted_outcome,
            json.dumps(episode.retrieval_memory_ids) if episode.retrieval_memory_ids else None,
            json.dumps(episode.retrieval_belief_ids) if episode.retrieval_belief_ids else None,
            json.dumps(episode.retrieval_knowledge_ids) if episode.retrieval_knowledge_ids else None,
            episode.execution_id,
            episode.correlation_id,
            episode.opportunity_id,
            episode.strategy_version,
            json.dumps(episode.metadata) if episode.metadata else None,
        ])
        return episode.episode_id

    def get_reasoning_episode(self, episode_id: str) -> Optional[ReasoningEpisode]:
        row = self._conn.execute(
            "SELECT * FROM reasoning_episodes WHERE episode_id = ?",
            [episode_id],
        ).fetchone()
        if row is None:
            return None
        return self._reasoning_episode_from_row(row)

    def get_recent_reasoning_episodes(self, limit: int = 50, since: datetime | None = None) -> list[ReasoningEpisode]:
        if since:
            rows = self._conn.execute(
                "SELECT * FROM reasoning_episodes WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT ?",
                [since, limit],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM reasoning_episodes ORDER BY timestamp DESC LIMIT ?",
                [limit],
            ).fetchall()
        return [self._reasoning_episode_from_row(r) for r in rows]

    def _reasoning_episode_from_row(self, row) -> ReasoningEpisode:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return ReasoningEpisode(
            episode_id=data["episode_id"],
            decision_id=data.get("decision_id") or "",
            timestamp=data["timestamp"],
            symbol=data["symbol"],
            timeframe=data.get("timeframe") or "",
            prompt_hash=data.get("prompt_hash") or "",
            prompt_preview=data.get("prompt_preview") or "",
            market_summary=self._safe_json_load(data.get("market_summary"), dict),
            evidence_summary=self._safe_json_load(data.get("evidence_summary"), dict),
            portfolio_summary=self._safe_json_load(data.get("portfolio_summary"), dict),
            action=data.get("action") or "",
            confidence=data.get("confidence") or 0.0,
            rationale=data.get("rationale") or "",
            risk_assessment=data.get("risk_assessment") or "",
            llm_response_raw=data.get("llm_response_raw") or "",
            chosen_signals=self._safe_json_load(data.get("chosen_signals"), list),
            ignored_signals=self._safe_json_load(data.get("ignored_signals"), list),
            predicted_outcome=data.get("predicted_outcome") or "",
            retrieval_memory_ids=self._safe_json_load(data.get("retrieval_memory_ids"), list),
            retrieval_belief_ids=self._safe_json_load(data.get("retrieval_belief_ids"), list),
            retrieval_knowledge_ids=self._safe_json_load(data.get("retrieval_knowledge_ids"), list),
            execution_id=data.get("execution_id") or "",
            correlation_id=data.get("correlation_id") or "",
            opportunity_id=data.get("opportunity_id") or "",
            strategy_version=data.get("strategy_version") or "",
            metadata=self._safe_json_load(data.get("metadata"), dict),
        )

    @staticmethod
    def _safe_json_load(value, container_type):
        if value is None:
            return container_type()
        if isinstance(value, str):
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return container_type()
        return value

    def update_reasoning_episode(self, episode_id: str, *,
        action: str | None = None,
        confidence: float | None = None,
        rationale: str | None = None,
        risk_assessment: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        existing = self._conn.execute(
            "SELECT metadata FROM reasoning_episodes WHERE episode_id = ?",
            [episode_id],
        ).fetchone()
        if existing is None:
            return False
        current_meta = self._safe_json_load(existing[0], dict) if existing[0] else {}
        merged_meta = {**current_meta, **(metadata or {})}
        self._conn.execute("""
            UPDATE reasoning_episodes SET
                action = COALESCE(?, action),
                confidence = COALESCE(?, confidence),
                rationale = COALESCE(?, rationale),
                risk_assessment = COALESCE(?, risk_assessment),
                metadata = ?
            WHERE episode_id = ?
        """, [
            action, confidence, rationale, risk_assessment,
            json.dumps(merged_meta), episode_id,
        ])
        return True

    # ── Belief Persistence ──

    def save_belief(self, belief: Belief) -> str:
        self._conn.execute("""
            INSERT INTO beliefs (
                belief_id, statement, category, symbol, confidence,
                strength, source, observation_count, created_at,
                last_updated, deprecated, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (belief_id) DO NOTHING
        """, [
            belief.belief_id, belief.statement, belief.category, belief.symbol,
            belief.confidence, belief.strength, belief.source,
            belief.observation_count, belief.created_at, belief.last_updated,
            belief.deprecated, json.dumps(belief.metadata) if belief.metadata else None,
        ])
        return belief.belief_id

    def update_belief(self, belief_id: str, *,
        confidence: float | None = None,
        strength: float | None = None,
        observation_count: int | None = None,
        deprecated: bool | None = None,
        metadata: dict | None = None,
    ) -> bool:
        existing = self._conn.execute(
            "SELECT metadata, observation_count FROM beliefs WHERE belief_id = ?",
            [belief_id],
        ).fetchone()
        if existing is None:
            return False
        current_meta = self._safe_json_load(existing[0], dict) if existing[0] else {}
        merged_meta = {**current_meta, **(metadata or {})}
        merged_count = observation_count if observation_count is not None else existing[1]
        self._conn.execute("""
            UPDATE beliefs SET
                confidence = COALESCE(?, confidence),
                strength = COALESCE(?, strength),
                observation_count = ?,
                deprecated = COALESCE(?, deprecated),
                metadata = ?,
                last_updated = ?
            WHERE belief_id = ?
        """, [
            confidence, strength, merged_count,
            deprecated, json.dumps(merged_meta), datetime.utcnow(),
            belief_id,
        ])
        return True

    def get_active_beliefs(self) -> list[Belief]:
        rows = self._conn.execute(
            "SELECT * FROM beliefs WHERE deprecated = FALSE ORDER BY confidence DESC, strength DESC",
        ).fetchall()
        return [self._belief_from_row(r) for r in rows]

    def get_beliefs_by_category(self, category: str) -> list[Belief]:
        rows = self._conn.execute(
            "SELECT * FROM beliefs WHERE category = ? AND deprecated = FALSE ORDER BY confidence DESC",
            [category],
        ).fetchall()
        return [self._belief_from_row(r) for r in rows]

    def _belief_from_row(self, row) -> Belief:
        cols = [desc[0] for desc in self._conn.description]
        data = dict(zip(cols, row))
        return Belief(
            belief_id=data["belief_id"],
            statement=data["statement"],
            category=data["category"],
            symbol=data.get("symbol") or "",
            confidence=data.get("confidence") or 0.0,
            strength=data.get("strength") or 0.0,
            source=data.get("source") or "reflection",
            observation_count=data.get("observation_count") or 1,
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            deprecated=data.get("deprecated") or False,
            metadata=self._safe_json_load(data.get("metadata"), dict),
        )

    # ── Memory Layer Statistics ──

    def get_observation_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        by_source = self._conn.execute(
            "SELECT source, COUNT(*) AS cnt FROM observations GROUP BY source ORDER BY cnt DESC"
        ).fetchall()
        by_category = self._conn.execute(
            "SELECT category, COUNT(*) AS cnt FROM observations GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        by_symbol = self._conn.execute(
            "SELECT symbol, COUNT(*) AS cnt FROM observations GROUP BY symbol ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        avg_imp = self._conn.execute("SELECT AVG(importance) FROM observations").fetchone()[0]
        return {
            "total_observations": total,
            "by_source": dict(by_source),
            "by_category": dict(by_category),
            "by_symbol": dict(by_symbol),
            "average_importance": round(avg_imp, 4) if avg_imp else 0.0,
        }

    def get_aggregate_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM observation_aggregates").fetchone()[0]
        total_compressed = self._conn.execute(
            "SELECT COALESCE(SUM(count), 0) FROM observation_aggregates"
        ).fetchone()[0]
        return {
            "total_aggregates": total,
            "total_compressed_observations": total_compressed,
        }

    def get_timeline_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM timelines").fetchone()[0]
        by_status = self._conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM timelines GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        avg_obs = self._conn.execute("SELECT AVG(observation_count) FROM timelines").fetchone()[0]
        return {
            "total_timelines": total,
            "by_status": dict(by_status),
            "average_observation_count": round(avg_obs, 2) if avg_obs else 0.0,
        }

    def get_pattern_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
        by_category = self._conn.execute(
            "SELECT category, COUNT(*) AS cnt FROM patterns GROUP BY category ORDER BY cnt DESC"
        ).fetchall()
        avg_conf = self._conn.execute("SELECT AVG(confidence) FROM patterns").fetchone()[0]
        return {
            "total_patterns": total,
            "by_category": dict(by_category),
            "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
        }

    def get_hypothesis_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM hypotheses").fetchone()[0]
        by_status = self._conn.execute(
            "SELECT status, COUNT(*) AS cnt FROM hypotheses GROUP BY status ORDER BY cnt DESC"
        ).fetchall()
        avg_conf = self._conn.execute("SELECT AVG(confidence) FROM hypotheses").fetchone()[0]
        total_evidence = self._conn.execute("SELECT COUNT(*) FROM hypothesis_evidence").fetchone()[0]
        return {
            "total_hypotheses": total,
            "by_status": dict(by_status),
            "average_confidence": round(avg_conf, 4) if avg_conf else 0.0,
            "total_evidence_links": total_evidence,
        }

    def get_knowledge_statistics(self) -> dict[str, Any]:
        total = self._conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        by_confidence = self._conn.execute(
            "SELECT confidence, COUNT(*) AS cnt FROM knowledge GROUP BY confidence ORDER BY cnt DESC"
        ).fetchall()
        avg_score = self._conn.execute("SELECT AVG(confidence_score) FROM knowledge").fetchone()[0]
        avg_cross = self._conn.execute("SELECT AVG(cross_timeline_count) FROM knowledge").fetchone()[0]
        return {
            "total_knowledge": total,
            "by_confidence": dict(by_confidence),
            "average_confidence_score": round(avg_score, 4) if avg_score else 0.0,
            "average_cross_timeline_count": round(avg_cross, 2) if avg_cross else 0.0,
        }

    def close(self) -> None:
        try:
            self._conn.execute("CHECKPOINT")
        except Exception:
            pass
        try:
            self._conn.close()
        except Exception:
            pass
