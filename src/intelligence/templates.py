from __future__ import annotations

# Linguistic templates — the ONLY source of worded descriptions.
# Every template uses named placeholders; the Formatter fills them.
# No engineering jargon, no raw metric names (pnl, iqr, cv).

ATR_TEMPLATE = "Typical {direction} movement: {value} ATR"

WINRATE_TEMPLATE = "Win rate: {pct}% ({count}/{total} trades)"

MAE_TEMPLATE = "Typical adverse movement: {value} ATR (IQR: {iqr})"

MFE_TEMPLATE = "Typical favorable movement: {value} ATR (IQR: {iqr})"

TIMING_TEMPLATE = "Median holding time: {value} bars (IQR: {iqr})"

PATTERN_TEMPLATE = (
    "During {field}={value}: {frequency}% frequency "
    "(confidence: {confidence})"
)

BIAS_TEMPLATE = "Data skew: {value} appears in {pct}% of samples"

INSUFFICIENT_TEMPLATE = (
    "Insufficient Historical Data — sample too small or quality "
    "too low for reliable analysis."
)

REPRESENTATIVES_TEMPLATE = "Representative experiences: {ids}"

EVIDENCE_HEADER = "Historical Market Context"
