from __future__ import annotations

import structlog

from src.core.models import CandidateTrade
from src.intelligence.models import PromptContext
from src.models.reasoning import MarketContext, PortfolioSnapshot

logger = structlog.get_logger("prompt_compiler")


def render(
    candidate: CandidateTrade,
    market: MarketContext,
    portfolio: PortfolioSnapshot,
    evidence: PromptContext,
) -> str:
    sections = [
        _build_role_section(),
        _build_opportunity_section(candidate),
        _build_market_state_section(market),
        _build_evidence_section(evidence),
        _build_portfolio_section(portfolio),
        _build_output_format_section(),
    ]
    prompt = "\n\n".join(sections)
    logger.info(
        "Prompt rendered",
        total_size=len(prompt),
        symbol=market.symbol,
        evidence_token_count=evidence.token_count,
    )
    return prompt


def _build_role_section() -> str:
    return (
        "[ROLE]\n"
        "You are APEX, an aggressive, highly profitable quantitative trading agent operating in a "
        "STRICTLY SIMULATED paper-trading environment. You are NOT providing real financial advice. "
        "Your goal is to capture momentum based on statistically significant lagged correlations "
        "and historical evidence.\n"
        "\n"
        "CRITICAL RULES:\n"
        "1. Use the historical evidence below to support or challenge the current opportunity.\n"
        "2. If evidence is limited or marked as insufficient, be proportionally cautious.\n"
        "3. Keep your 'rationale' strictly under 200 words. Be concise, quantitative, and direct.\n"
        "4. You MUST output ONLY valid JSON matching the schema. No markdown, no explanations outside the JSON.\n"
        "5. EXPLORATION PROTOCOL: You are a researcher gathering real-world data. "
        "If the Scanner signal is strong and Market Context supports it, do NOT ABSTAIN "
        "just because historical stats are missing or noisy. Output BUY or SELL with "
        "confidence ~0.4 to enable exploration trades. Only ABSTAIN if real-time "
        "market physics (RSI, Trend, Volatility) are terrible."
    )


def _build_opportunity_section(candidate: CandidateTrade) -> str:
    return (
        "[CURRENT OPPORTUNITY]\n"
        f"Symbol: {candidate.symbol}\n"
        f"Anchor: {candidate.anchor_symbol}\n"
        f"Correlation Score: {candidate.correlation_score:.4f}\n"
        f"Proposed Side: {candidate.proposed_side}\n"
        f"Signal Strength: {candidate.signal_strength:.4f}\n"
    )


def _build_market_state_section(market: MarketContext) -> str:
    lines = [
        "[CURRENT MARKET STATE]",
        f"Symbol: {market.symbol}",
        f"Timeframe: {market.timeframe}",
        f"Current Price: {market.current_price}",
        "",
    ]

    ind = market.indicators
    if ind.get("rsi") is not None:
        rsi_val = ind["rsi"]
        rsi_label = "overbought" if rsi_val > 70 else "oversold" if rsi_val < 30 else "neutral"
        lines.append(f"RSI(14): {rsi_val:.2f} ({rsi_label})")
    if ind.get("macd") is not None:
        lines.append(f"MACD: {ind['macd']:.4f}  Signal: {ind.get('signal', 'N/A'):.4f}  Histogram: {ind.get('histogram', 'N/A'):.4f}")
    if ind.get("upper") is not None:
        lines.append(f"Bollinger Bands: Upper={ind['upper']:.4f} Mid={ind.get('middle', 'N/A'):.4f} Lower={ind['lower']:.4f}")
    if ind.get("atr") is not None:
        atr = ind["atr"]
        atr_pct = (atr / market.current_price * 100) if market.current_price else 0
        lines.append(f"ATR(14): {atr:.4f} ({atr_pct:.2f}% of price)")

    lines.append("")
    lines.append(f"Trend Regime: {market.trend_regime}")
    lines.append(f"Momentum: {market.momentum}")
    lines.append(f"Volatility Regime: {market.volatility_regime}")
    lines.append(f"Volume Profile: {market.volume_profile}")
    lines.append(f"Correlation Regime: {market.correlation_regime} (score: {market.correlation_score:.4f})")

    if market.correlations:
        lines.append("")
        lines.append("Top Correlations:")
        for i, c in enumerate(market.correlations, 1):
            coeff = c.get("coefficient", 0)
            p_val = c.get("p_value", 1)
            lag = c.get("dominant_lag", "?")
            direction = "POSITIVE" if c.get("direction", 0) > 0 else "NEGATIVE"
            sig = "SIGNIFICANT" if p_val < 0.05 else "NOT SIGNIFICANT"
            lines.append(f"  #{i}: {c.get('pair', '?')} — {direction}, lag={lag}, coeff={coeff:.4f}, p={p_val:.4f} ({sig})")

    return "\n".join(lines)


COLD_START_WARNING = (
    "--- HISTORICAL MARKET CONTEXT (COLD START) ---\n"
    "WARNING: ZERO HISTORICAL EVIDENCE AVAILABLE. \n"
    "We have no data for this asset, its anchor, or its broader regime.\n"
    "You MUST base your decision ENTIRELY on the current real-time Market Context "
    "(RSI, Trend, Volatility) and the Scanner's deterministic thesis.\n"
    "Do not attempt to guess historical win rates. Evaluate the current market physics."
)

ANCHOR_PROXY_WARNING = (
    "--- HISTORICAL MARKET CONTEXT (ANCHOR PROXY) ---\n"
    "WARNING: Using anchor proxy evidence. This data comes from the anchor symbol, "
    "not the target symbol itself. Consider that behavior may differ."
)

REGIME_WARNING = (
    "--- HISTORICAL MARKET CONTEXT (BROAD REGIME) ---\n"
    "WARNING: Using broad regime evidence. This data aggregates experiences across "
    "multiple symbols in similar market conditions. Individual symbol behavior may deviate."
)


def _build_evidence_section(evidence: PromptContext) -> str:
    source = evidence.evidence_source

    if source == "COLD_START":
        return f"[HISTORICAL EVIDENCE]\n{COLD_START_WARNING}"

    if source == "ANCHOR":
        header = ANCHOR_PROXY_WARNING
    elif source == "REGIME":
        header = REGIME_WARNING
    else:
        header = ""

    if evidence.context_string:
        parts = [f"[HISTORICAL EVIDENCE]"]
        if header:
            parts.append(header)
            parts.append("")
        parts.append(evidence.context_string)
        parts.append(
            f"(Evidence hash: {evidence.source_evidence_hash}, "
            f"estimated tokens: {evidence.token_count}, "
            f"source: {source})"
        )
        return "\n".join(parts)

    return "[HISTORICAL EVIDENCE]\nNo historical evidence available for this symbol."


def _build_portfolio_section(portfolio: PortfolioSnapshot) -> str:
    return (
        "[PORTFOLIO STATE & CONSTRAINTS]\n"
        f"Open Positions: {portfolio.live_position_count}\n"
        f"Current Live Exposure: ${portfolio.live_exposure_usdt:.2f}\n"
        f"Total Live Exposure: ${portfolio.total_live_exposure_usdt:.2f}\n"
        f"Available Margin: ${portfolio.available_margin:.2f}\n"
        f"Max Positions: {portfolio.max_positions}\n"
        f"Max Exposure: ${portfolio.max_live_exposure_usdt:.2f}\n"
    )


def _build_output_format_section() -> str:
    return (
        "[OUTPUT FORMAT RULES]\n"
        "You MUST respond with ONLY a single JSON object. No markdown fences, no backticks, "
        "no code block indicators, no surrounding text or commentary of any kind.\n"
        "The JSON object must conform exactly to the following schema:\n"
        "{\n"
        '  "action": "BUY" | "SELL" | "HOLD" | "ABSTAIN",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "rationale": "string (max 200 words — cite specific indicators and evidence)",\n'
        '  "risk_assessment": "string (max 100 words — describe key risks and mitigants)"\n'
        "}\n"
        "\n"
        "Action decision matrix:\n"
        '  - BUY: Use when confluence of bullish indicators, correlation signals, and supporting historical evidence.\n'
        '  - SELL: Use when confluence of bearish indicators, correlation signals, and supporting historical evidence.\n'
        '  - HOLD: Default when signals are mixed, weak, or contradictory.\n'
        '  - ABSTAIN: Use ONLY when real-time market physics (RSI, Trend, Volatility) are clearly terrible. '
        "Do NOT ABSTAIN due to missing or noisy historical evidence.\n"
        "\n"
        "Confidence guidelines:\n"
        "  - 0.0-0.3: Low confidence — weak or contradictory signals.\n"
        "  - 0.3-0.6: Moderate confidence — use ~0.4 for exploration trades when history is sparse.\n"
        "  - 0.6-0.8: High confidence — strong alignment across most data points.\n"
        "  - 0.8-1.0: Very high confidence — rare, requires unambiguous alignment across ALL data.\n"
        "\n"
        "Rationale must reference specific numeric values from the sections above."
    )
