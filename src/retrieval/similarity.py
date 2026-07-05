from __future__ import annotations

from typing import Any, Optional

from src.learning.feature_catalog import FeatureCatalog
from src.retrieval.models import Explanation, SimilarityBreakdown, RetrievalRecord
from src.retrieval.weights import SimilarityWeights


def _normalized_absolute_distance(
    query_val: Optional[float],
    candidate_val: Optional[float],
    feature_range: float,
) -> float:
    if query_val is None and candidate_val is None:
        return 1.0
    if query_val is None or candidate_val is None:
        return 0.5
    diff = abs(query_val - candidate_val)
    raw = 1.0 - min(diff / feature_range, 1.0)
    return max(0.0, min(1.0, raw))


def _exact_match(
    query_val: Any,
    candidate_val: Any,
) -> float:
    if query_val is None and candidate_val is None:
        return 1.0
    if query_val is None or candidate_val is None:
        return 0.0
    return 1.0 if query_val == candidate_val else 0.0


FEATURE_GROUP_MARKET = [
    "market.normalized_entry_atr_multiple",
    "market.normalized_exit_atr_multiple",
    "market.entry_rsi_percentile",
    "market.entry_volatility_percentile",
    "market.trend_regime",
    "market.volatility_regime",
]

FEATURE_GROUP_EXECUTION = [
    "execution.total_slippage_bps",
    "execution.total_fees_bps",
]

FEATURE_GROUP_RISK = [
    "risk.initial_risk_atr_multiple",
    "risk.realized_rr",
]

FEATURE_GROUP_OUTCOME = [
    "outcome.pnl_atr_multiple",
    "outcome.mfe_atr_multiple",
    "outcome.mae_atr_multiple",
]

FEATURE_GROUP_CONTEXT = [
    "context.symbol",
    "context.timeframe",
    "market.correlation_regime",
]

GROUP_NAMES = {
    "market": FEATURE_GROUP_MARKET,
    "execution": FEATURE_GROUP_EXECUTION,
    "risk": FEATURE_GROUP_RISK,
    "outcome": FEATURE_GROUP_OUTCOME,
    "context": FEATURE_GROUP_CONTEXT,
}


def _record_to_feature_dict(record: RetrievalRecord) -> dict[str, Any]:
    return {
        "market.normalized_entry_atr_multiple": record.normalized_entry_atr_multiple,
        "market.normalized_exit_atr_multiple": record.normalized_exit_atr_multiple,
        "market.entry_rsi_percentile": record.entry_rsi_percentile,
        "market.entry_volatility_percentile": record.entry_volatility_percentile,
        "market.trend_regime": record.trend_regime,
        "market.volatility_regime": record.volatility_regime,
        "market.correlation_regime": record.correlation_regime,
        "execution.total_slippage_bps": record.total_slippage_bps,
        "execution.total_fees_bps": record.total_fees_bps,
        "risk.initial_risk_atr_multiple": record.initial_risk_atr_multiple,
        "risk.realized_rr": record.realized_rr,
        "outcome.pnl_atr_multiple": record.pnl_atr_multiple,
        "outcome.mfe_atr_multiple": record.mfe_atr_multiple,
        "outcome.mae_atr_multiple": record.mae_atr_multiple,
        "context.symbol": record.symbol,
        "context.timeframe": record.timeframe,
    }


class SimilarityEngine:
    """Deterministic similarity computation using FeatureCatalog metadata.
    Each category computes an independent score.
    Builds structured Explanation objects for every group.
    No AI, no embeddings, no learned weights."""

    def __init__(
        self,
        feature_catalog: FeatureCatalog,
        weights: SimilarityWeights | None = None,
    ):
        self._catalog = feature_catalog
        self._weights = (weights or SimilarityWeights()).to_dict()

    def compute_similarity(
        self,
        query_proj: dict[str, Any],
        candidate: RetrievalRecord,
    ) -> SimilarityBreakdown:
        candidate_features = _record_to_feature_dict(candidate)

        results: dict[str, Any] = {}
        explanations: list[Explanation] = []

        for group_name in ["market", "execution", "risk", "outcome", "context"]:
            feature_ids = GROUP_NAMES[group_name]
            group_result, group_explanation = self._score_group(
                feature_ids, query_proj, candidate_features, group_name,
            )
            results[group_name] = group_result
            explanations.append(group_explanation)

        overall = (
            self._weights.get("market", 0.35) * results["market"]["score"]
            + self._weights.get("context", 0.25) * results["context"]["score"]
            + self._weights.get("risk", 0.20) * results["risk"]["score"]
            + self._weights.get("execution", 0.10) * results["execution"]["score"]
            + self._weights.get("outcome", 0.10) * results["outcome"]["score"]
        )

        return SimilarityBreakdown(
            market_score=results["market"]["score"],
            execution_score=results["execution"]["score"],
            risk_score=results["risk"]["score"],
            outcome_score=results["outcome"]["score"],
            context_score=results["context"]["score"],
            overall_score=round(overall, 6),
            explanations=explanations,
            group_details={
                name: results[name]
                for name in ["market", "execution", "risk", "outcome", "context"]
            },
        )

    def _score_group(
        self,
        feature_ids: list[str],
        query_features: dict[str, Any],
        candidate_features: dict[str, Any],
        group_name: str,
    ) -> tuple[dict[str, Any], Explanation]:
        scores: list[float] = []
        matched: list[str] = []
        mismatched: list[str] = []
        ignored: list[str] = []
        missing: list[str] = []
        feature_scores: dict[str, float] = {}
        textual: list[str] = []

        for fid in feature_ids:
            if fid not in query_features:
                if fid in candidate_features and candidate_features[fid] is not None:
                    ignored.append(fid)
                    textual.append(f"{fid.split('.')[-1]}: ignored (not in query)")
                elif fid in candidate_features and candidate_features[fid] is None:
                    missing.append(fid)
                    textual.append(f"{fid.split('.')[-1]}: missing (both None)")
                continue

            qv = query_features[fid]
            cv = candidate_features.get(fid)
            feat_def = self._catalog.get_feature(fid)

            if feat_def is None:
                continue

            comp_type = feat_def.comparison_type
            comp_range = feat_def.comparison_range
            distance_fn = feat_def.distance_function

            if distance_fn == "exact_match" or comp_type == "categorical":
                score = _exact_match(qv, cv)
            else:
                score = _normalized_absolute_distance(qv, cv, comp_range)

            scores.append(score)
            feature_scores[fid] = round(score, 6)
            short = fid.split(".")[-1]

            if score > 0.8:
                matched.append(fid)
                textual.append(f"{short}: matched ({score:.3f})")
            elif score < 0.2:
                mismatched.append(fid)
                textual.append(f"{short}: mismatched ({score:.3f})")
            else:
                textual.append(f"{short}: partial ({score:.3f})")

        group_score = round(sum(scores) / len(scores), 6) if scores else 0.0
        total_evaluated = len(matched) + len(mismatched)
        confidence = round(total_evaluated / (total_evaluated + len(ignored) + len(missing)), 6) if (total_evaluated + len(ignored) + len(missing)) > 0 else 0.0

        explanation = Explanation(
            group_name=group_name,
            matched_features=sorted(matched),
            mismatched_features=sorted(mismatched),
            ignored_features=sorted(ignored),
            missing_features=sorted(missing),
            confidence_score=confidence,
        )

        group_result = {
            "score": group_score,
            "features": dict(sorted(feature_scores.items())),
            "explanations": textual,
            "count": len(scores),
        }

        return group_result, explanation
