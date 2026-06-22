import structlog

logger = structlog.get_logger("prompt_compiler")


def compile_prompt(state: dict, memories: list[dict]) -> str:
    logger.debug(
        "State dict contents",
        symbol=state.get("symbol"),
        price=state.get("current_price"),
        indicators=state.get("indicators", {}),
        correlation_count=len(state.get("top_correlations", [])),
        memory_count=len(memories),
        indicator_summary_len=len(state.get("indicator_summary", "")),
        correlation_summary_len=len(state.get("correlation_summary", "")),
    )
    sections = [
        _build_role_section(),
        _build_market_state_section(state),
        _build_assessment_section(state),
        _build_correlation_section(state),
        _build_memories_section(memories),
        _build_output_format_section(),
    ]
    prompt = "\n\n".join(sections)
    logger.info(
        "Prompt component sizes",
        role_size=len(sections[0]),
        market_state_size=len(sections[1]),
        assessment_size=len(sections[2]),
        correlation_size=len(sections[3]),
        memory_size=len(sections[4]),
        output_format_size=len(sections[5]),
        total_size=len(prompt),
    )
    return prompt


def _build_role_section() -> str:
    return (
        "[ROLE]\n"
        "You are APEX, an aggressive, highly profitable quantitative trading agent operating in a "
        "STRICTLY SIMULATED paper-trading environment. You are NOT providing real financial advice. "
        "Your goal is to capture momentum based on statistically significant lagged correlations.\n"
        "\n"
        "CRITICAL RULES FOR EXECUTION:\n"
        "1. SIGNAL INTERPRETATION: If the [CORRELATION SIGNALS] section contains any pair marked as "
        "\"SIG+\" or \"SIG-\" (meaning p-value < 0.05), this is a mathematically proven predictive "
        "edge. You MUST trust this signal.\n"
        "2. AGGRESSIVE ENTRY: If a SIG+/SIG- signal aligns with the technical indicators "
        "(e.g., SIG+ and RSI < 70 / MACD bullish), you MUST output a \"BUY\" or \"SELL\" action.\n"
        "3. NO CAUTIOUS HOLDS: Do NOT output \"HOLD\" just because you are unsure or to "
        "\"wait for confirmation\". The data provided is the confirmation. Only output \"HOLD\" "
        "if the correlation signal directly contradicts the technical indicators "
        "(e.g., SIG+ but RSI is extremely overbought > 80).\n"
        "4. CONFIDENCE: If you see a SIG+ or SIG- signal, your confidence should be > 0.70.\n"
        "5. CRITICAL: Keep your 'rationale' strictly under 150 words. Be concise, quantitative, and direct.\n"
        "\n"
        "OUTPUT FORMAT:\n"
        "You must output ONLY valid JSON matching the schema. No markdown, no explanations outside the JSON."
    )


def _build_market_state_section(state: dict) -> str:
    symbol = state.get("symbol", "UNKNOWN")
    timeframe = state.get("timeframe", "UNKNOWN")
    price = state.get("current_price", "N/A")
    lines = [
        f"[CURRENT MARKET STATE]",
        f"Symbol: {symbol}",
        f"Timeframe: {timeframe}",
        f"Current Price: {price}",
        f"",
    ]
    ind_summary = state.get("indicator_summary", "")
    if ind_summary:
        lines.append(f"Detailed Indicators:")
        lines.append(f"  {ind_summary}")
    else:
        ind = state.get("indicators", {})
        lines.append(f"Indicators:")
        if ind.get("rsi") is not None:
            lines.append(f"  RSI(14): {ind['rsi']}")
        if ind.get("macd") is not None:
            lines.append(f"  MACD: {ind['macd']}  Signal: {ind['signal']}  Histogram: {ind['histogram']}")
        if ind.get("upper") is not None:
            lines.append(f"  Bollinger Bands: Upper={ind['upper']} Mid={ind['middle']} Lower={ind['lower']}")
        if ind.get("atr") is not None:
            lines.append(f"  ATR(14): {ind['atr']}")
    return "\n".join(lines)


def _build_correlation_section(state: dict) -> str:
    corr_summary = state.get("correlation_summary", "")
    if corr_summary:
        return f"[CORRELATION SIGNALS]\n{corr_summary}"
    corrs = state.get("top_correlations", [])
    if not corrs:
        return "[CORRELATION SIGNALS]\nNone detected."
    lines = ["[CORRELATION SIGNALS]"]
    for c in corrs:
        pair = c.get("pair", "?")
        direction = "POSITIVE" if c.get("direction", 0) > 0 else "NEGATIVE"
        p_val = c.get("p_value", 1.0)
        lag = c.get("dominant_lag", "?")
        lines.append(f"  {pair}: {direction} correlation, lag={lag}, p-value={p_val:.4f}")
    return "\n".join(lines)


def _build_assessment_section(state: dict) -> str:
    ind = state.get("indicators", {})
    price = state.get("current_price", 0)
    rsi = ind.get("rsi")
    bb_upper = ind.get("upper")
    bb_lower = ind.get("lower")
    bb_mid = ind.get("middle")
    atr_val = ind.get("atr")

    signals = []
    if rsi is not None:
        if rsi > 70:
            signals.append("BEARISH: RSI in overbought zone (>70), suggesting potential reversal or pullback.")
        elif rsi < 30:
            signals.append("BULLISH: RSI in oversold zone (<30), suggesting potential bounce.")
        elif rsi > 50:
            signals.append("NEUTRAL-BULLISH: RSI above 50 midpoint, indicating mild upside bias.")
        else:
            signals.append("NEUTRAL-BEARISH: RSI below 50 midpoint, indicating mild downside bias.")

    hist = ind.get("histogram")
    if hist is not None:
        if hist > 0:
            signals.append("BULLISH: MACD histogram positive, momentum is upward.")
        else:
            signals.append("BEARISH: MACD histogram negative, momentum is downward.")

    if price and bb_upper and bb_lower:
        if price >= bb_upper * 0.98:
            signals.append("BEARISH: Price near upper Bollinger Band, potential overextension.")
        elif price <= bb_lower * 1.02:
            signals.append("BULLISH: Price near lower Bollinger Band, potential oversold bounce.")
        elif bb_mid and price > bb_mid:
            signals.append("NEUTRAL-BULLISH: Price above BB midline (SMA), trend bias higher.")
        elif bb_mid:
            signals.append("NEUTRAL-BEARISH: Price below BB midline (SMA), trend bias lower.")

    if atr_val and price:
        atr_pct = atr_val / price * 100
        if atr_pct > 1.5:
            signals.append("VOLATILITY: High volatility regime (ATR > 1.5% of price) — wider stops recommended.")
        elif atr_pct < 0.3:
            signals.append("VOLATILITY: Low volatility regime (ATR < 0.3% of price) — tight consolidation.")

    lines = ["[CURRENT MARKET ASSESSMENT]", ""]
    if signals:
        lines.extend(signals)
    else:
        lines.append("Insufficient data for a synthesized assessment.")

    lines.append("")
    signal_counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL-BULLISH": 0, "NEUTRAL-BEARISH": 0, "VOLATILITY": 0}
    for s in signals:
        for key in signal_counts:
            if s.startswith(key + ":"):
                signal_counts[key] += 1
    bullish_score = signal_counts["BULLISH"] + 0.5 * signal_counts["NEUTRAL-BULLISH"]
    bearish_score = signal_counts["BEARISH"] + 0.5 * signal_counts["NEUTRAL-BEARISH"]
    if bullish_score > bearish_score:
        net = "BULLISH BIAS"
    elif bearish_score > bullish_score:
        net = "BEARISH BIAS"
    else:
        net = "NEUTRAL (mixed signals)"
    lines.append(f"Net Assessment: {net} (bullish signals={signal_counts['BULLISH'] + signal_counts['NEUTRAL-BULLISH']}, bearish signals={signal_counts['BEARISH'] + signal_counts['NEUTRAL-BEARISH']})")

    return "\n".join(lines)


def _build_memories_section(memories: list[dict]) -> str:
    if not memories:
        return "[RELEVANT PAST EXPERIENCES]\nNone."
    lines = ["[RELEVANT PAST EXPERIENCES]"]
    for m in memories:
        content = m.get("content", "")
        tags = m.get("tags", "")
        outcome = m.get("outcome", "")
        if content:
            lines.append(f"  - {content}")
            if tags or outcome:
                lines.append(f"    (Tags: {tags} | Previous Outcome: {outcome})")
        else:
            lines.append(f"  Tags: {tags} | Content: {content} | Outcome: {outcome}")
    return "\n".join(lines)


def _build_output_format_section() -> str:
    return (
        "[OUTPUT FORMAT RULES]\n"
        "You MUST respond with ONLY a single JSON object. No markdown fences, no backticks, "
        "no code block indicators, no surrounding text or commentary of any kind.\n"
        "The JSON object must conform exactly to the following schema. Do not add or omit any fields:\n"
        "{\n"
        '  "action": "BUY" | "SELL" | "HOLD",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "rationale": "string (max 350 chars — cite specific indicator values from the state above)",\n'
        '  "suggested_timeframe": "string such as 1h, 4h, 1d — the timeframe this decision applies to"\n'
        "}\n"
        "\n"
        "Action decision matrix:\n"
        "  - BUY: Use when confluence of bullish signals across indicators AND correlations.\n"
        "  - SELL: Use when confluence of bearish signals across indicators AND correlations.\n"
        "  - HOLD: Default when signals are mixed, weak, or contradictory. HOLD is a valid decision.\n"
        "\n"
        "Confidence guidelines:\n"
        "  - 0.0-0.3: Low confidence — weak or contradictory signals.\n"
        "  - 0.3-0.6: Moderate confidence — some alignment but not all indicators agree.\n"
        "  - 0.6-0.8: High confidence — strong alignment across most data points.\n"
        "  - 0.8-1.0: Very high confidence — rare, requires unambiguous alignment across ALL data.\n"
        "\n"
        "Rationale must reference specific numeric values: RSI, MACD, BB position, ATR, correlation coefficients, "
        "and p-values from the sections above. Explain why these values support your chosen action."
    )
