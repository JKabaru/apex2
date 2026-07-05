from __future__ import annotations

from src.intelligence.formatter import ExperienceEvidenceFormatter
from src.intelligence.models import (
    BiasSummary,
    EvidenceProvenance,
    ExperienceEvidence,
    Pattern,
    PromptContext,
    RepresentativeExperience,
)


def test_format_includes_header():
    evidence = ExperienceEvidence(
        sample_size=47,
        evidence_quality="HIGH",
        is_sufficient=True,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "Historical Market Context" in ctx.context_string


def test_format_mae_template():
    evidence = ExperienceEvidence(
        sample_size=47,
        evidence_quality="HIGH",
        is_sufficient=True,
        median_mae_atr=0.6,
        pnl_iqr=0.4,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "Typical adverse movement: 0.6 ATR" in ctx.context_string
    assert "IQR" in ctx.context_string


def test_format_mfe_template():
    evidence = ExperienceEvidence(
        sample_size=47,
        evidence_quality="HIGH",
        is_sufficient=True,
        median_mfe_atr=1.8,
        pnl_iqr=0.4,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "Typical favorable movement: 1.8 ATR" in ctx.context_string


def test_format_winrate():
    evidence = ExperienceEvidence(
        sample_size=30,
        evidence_quality="HIGH",
        is_sufficient=True,
        win_rate_pct=64.0,
        median_pnl_atr=0.82,
        overall_confidence=0.85,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "Win rate" in ctx.context_string
    assert "64" in ctx.context_string
    assert "0.82" in ctx.context_string
    assert "HIGH" in ctx.context_string


def test_format_patterns_no_raw_metrics():
    evidence = ExperienceEvidence(
        sample_size=30,
        evidence_quality="HIGH",
        is_sufficient=True,
        success_patterns=[
            Pattern(
                field="trend_regime",
                value="BULLISH",
                frequency=0.82,
                confidence_score=0.9,
            ),
        ],
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "82" in ctx.context_string or "82.0" in ctx.context_string
    assert "trend_regime" in ctx.context_string
    assert "BULLISH" in ctx.context_string
    assert "HIGH" in ctx.context_string  # confidence label


def test_format_pattern_low_confidence():
    evidence = ExperienceEvidence(
        sample_size=10,
        evidence_quality="LOW",
        is_sufficient=True,
        success_patterns=[
            Pattern(
                field="volatility_regime",
                value="HIGH",
                frequency=0.75,
                confidence_score=0.3,
            ),
        ],
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "LOW" in ctx.context_string


def test_format_bias():
    evidence = ExperienceEvidence(
        sample_size=50,
        evidence_quality="MEDIUM",
        is_sufficient=True,
        bias_summary=BiasSummary(
            symbol_distribution={"BTCUSDT": 40, "ETHUSDT": 10},
        ),
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "BTCUSDT" in ctx.context_string
    assert "80" in ctx.context_string  # 40/50 * 100


def test_format_representatives():
    evidence = ExperienceEvidence(
        sample_size=10,
        evidence_quality="MEDIUM",
        is_sufficient=True,
        representatives=[
            RepresentativeExperience(
                experience_id="exp-1",
                similarity_score=0.95,
                why_selected="test",
            ),
        ],
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "exp-1" in ctx.context_string


def test_format_timing():
    evidence = ExperienceEvidence(
        sample_size=20,
        evidence_quality="HIGH",
        is_sufficient=True,
        median_duration_bars=16.0,
        duration_iqr=10.0,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert "bars" in ctx.context_string
    assert "16" in ctx.context_string


def test_format_no_engineering_jargon():
    evidence = ExperienceEvidence(
        sample_size=47,
        evidence_quality="HIGH",
        is_sufficient=True,
        median_pnl_atr=0.82,
        median_mae_atr=0.6,
        median_mfe_atr=1.8,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    # These raw metric names must NOT appear in the output
    assert "pnl_atr" not in ctx.context_string
    assert "mae_atr" not in ctx.context_string
    assert "mfe_atr" not in ctx.context_string
    assert "cv" not in ctx.context_string
    assert "percentile" not in ctx.context_string
    assert "similarity_score" not in ctx.context_string


def test_prompt_context_metadata():
    evidence = ExperienceEvidence(
        sample_size=30,
        evidence_quality="HIGH",
        is_sufficient=True,
        win_rate_pct=64.0,
        median_pnl_atr=0.82,
    )
    formatter = ExperienceEvidenceFormatter()
    ctx = formatter.format(evidence)
    assert isinstance(ctx, PromptContext)
    assert len(ctx.section_order) > 0
    assert ctx.template_version == "1.0"
    assert len(ctx.source_evidence_hash) > 0
    assert ctx.token_count > 0
