from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Optional

import structlog

from src.evaluation.storage import EvaluationCorpus
from src.llm.registry import LLMRegistry
from src.research.storage import ResearchCorpusStore

logger = structlog.get_logger("adaptive_feedback")

DEFAULT_CONFIG = {
    "auto_merge": True,
    "show_confidence_in_prompt": True,
    "evidence_min_count": 1,
}

COOLDOWN_CYCLES = 2

SENSITIVE_CONFIG_PATHS = {"binance", "llm"}

_RESEARCH_PARAM_BOUNDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "correlation.rolling_window_candles": (100, 5000),
    "correlation.max_lag": (1, 100),
    "correlation.base_half_life": (10, 500),
    "correlation.min_half_life": (5, 100),
    "correlation.max_half_life": (30, 1000),
    "correlation.acf_truncation_lag": (1, 50),
    "correlation.alpha_crit": (0.001, 0.1),
    "correlation.update_buffer_candles": (1, 100),
    "research.pattern_high_conf_threshold": (0.0, 1.0),
    "research.pattern_overconfidence_high_gap": (0.0, 1.0),
    "research.pattern_overconfidence_medium_gap": (0.0, 1.0),
    "research.pattern_underconfidence_wr": (0.0, 1.0),
    "research.pattern_side_imbalance": (0.0, 1.0),
    "research.pattern_stop_loss_rate": (0.0, 1.0),
    "research.pattern_short_hold_wr": (0.0, 1.0),
    "research.pattern_min_evals": (1, 1000),
    "research.pattern_min_high_conf_evals": (1, 1000),
    "research.pattern_min_low_conf_evals": (1, 1000),
    "research.pattern_min_side_evals": (1, 100),
    "research.pattern_min_tier_evals": (1, 1000),
    "research.pattern_min_duration_evals": (1, 100),
    "research.min_sample_size": (1, 1000),
    "research.min_intervention_evals": (1, 100),
    "research.min_simulation_evals": (1, 100),
    "research.min_effect_size": (0.001, 1.0),
    "research.min_improvement": (0.0, 1.0),
    "research.cycle_interval": (1, 100),
    "research.metric_min_subgroup": (1, 100),
    "research.metric_min_losses": (1, 50),
    "research.observation_calibration_drift_threshold": (0.0, 1.0),
    "research.observation_calibration_severe_threshold": (0.0, 1.0),
    "research.observation_calibration_min_sample": (1, 1000),
    "research.observation_regime_min_sample": (1, 1000),
    "research.observation_low_win_rate": (0.0, 1.0),
    "research.observation_high_win_rate": (0.0, 1.0),
    "research.observation_small_sample_threshold": (1, 10000),
    "evidence.cold_start.max_confidence": (0.0, 1.0),
    "evidence.regime.max_confidence": (0.0, 1.0),
    "evidence.anchor.max_confidence": (0.0, 1.0),
    "evidence.exact.max_confidence": (0.0, 1.0),
    "evidence.exact_threshold": (0.0, 0.6),
    "scanner.min_correlation": (0.0, 1.0),
    "scanner.max_p_value": (0.0, 1.0),
    "learning.evidence_min_count": (1, None),
    "adaptive.auto_merge": (None, None),
    "adaptive.show_confidence_in_prompt": (None, None),
    "adaptive_feedback.llm_interval_cycles": (1, 100),
    "adaptive_feedback.reflection_interval_seconds": (30, 3600),
}

_EXECUTION_RISK_BOUNDS: dict[str, tuple[Optional[float], Optional[float]]] = {
    "execution.leverage": (1, 125),
    "execution.sizing_value": (0.1, 100.0),
    "execution.stop_loss_pct": (0.90, 0.99),
    "execution.take_profit_pct": (1.01, 2.0),
    "execution.spread_bps": (0.0, 10.0),
    "execution.fee_bps": (0.0, 10.0),
    "execution.slippage_bps": (0.0, 10.0),
    "execution.trailing_stop_atr_mult": (0.5, 5.0),
    "execution.max_risk_pct": (0.001, 0.1),
    "risk.max_positions": (1, 10),
    "risk.max_live_exposure_usdt": (100.0, 1_000_000.0),
    "risk.min_llm_confidence": (0.0, 1.0),
}

_ALL_BOUNDS = {**_RESEARCH_PARAM_BOUNDS, **_EXECUTION_RISK_BOUNDS}


class AdaptiveFeedbackEngine:
    """Autonomous threshold adjustment engine.

    Modes:
      "deterministic" — hardcoded rules using reflection metrics.
      "llm"           — LLM meta-cognition prompt, falls back to deterministic on failure.
      "both"          — LLM first, then deterministic fills anything the LLM didn't change.
    """

    def __init__(
        self,
        config_store: Any,
        llm_registry: Optional[LLMRegistry] = None,
        llm_model: str = "",
        evaluation_corpus: Optional[EvaluationCorpus] = None,
        research_store: Optional[ResearchCorpusStore] = None,
    ):
        self._config_store = config_store
        self._llm_registry = llm_registry
        self._llm_model = llm_model
        self._eval_corpus = evaluation_corpus
        self._research_store = research_store
        self._cycle_count = 0
        self._healthy_cycles = 0
        self._cooldowns: dict[str, int] = {}

    # ── public API ───────────────────────────────────────────────────────

    async def run(
        self,
        mode: str,
        config: dict,
        observations: list[dict],
        beliefs: list[Any],
    ) -> dict[str, Any]:
        """Run the feedback loop. Returns adjustments dict (empty if no changes)."""
        self._cycle_count += 1
        adjustments: dict[str, Any] = {}

        if mode in ("deterministic", "both"):
            det = self._run_deterministic(config, observations)
            adjustments.update(det)

        if mode in ("llm", "both"):
            llm_adj = await self._run_llm_reflection(config, observations, beliefs)
            for k, v in llm_adj.items():
                if k not in adjustments:
                    adjustments[k] = v

        if not adjustments:
            return {}

        self._apply_cooldowns(adjustments)
        self._persist(adjustments, config, observations)
        return adjustments

    def load_state(self) -> dict:
        """Load saved feedback state from config_store. Returns dict of values."""
        return self._config_store.load_feedback_state() or {}

    # ── deterministic rules ─────────────────────────────────────────────

    def _run_deterministic(self, config: dict, observations: list[dict]) -> dict:
        adj: dict[str, Any] = {}
        metrics = self._extract_metrics(observations)

        overturn_rate = metrics.get("overturn_rate", 0.0)
        low_conf_ratio = metrics.get("low_conf_ratio", 0.0)
        pending_count = metrics.get("pending_count", 0)
        adaptation_count = metrics.get("adaptation_count", 0)
        total_trades = metrics.get("total_trades", 0)

        current_merge = config.get("adaptive", {}).get("auto_merge", True)
        current_confidence = config.get("adaptive", {}).get("show_confidence_in_prompt", True)
        current_evidence = config.get("learning", {}).get("evidence_min_count", 1)

        rule_triggers: list[str] = []

        # Rule 1: High overturn rate — toggle show_confidence
        if total_trades >= 3 and overturn_rate > 0.30:
            if self._allowed("show_confidence_in_prompt"):
                adj["show_confidence_in_prompt"] = not current_confidence
                rule_triggers.append(
                    f"overturn_rate={overturn_rate:.2f} > 0.30 → toggle show_confidence"
                )

        # Rule 2: Too many low-confidence actions — decrease min_llm_confidence
        if total_trades >= 3 and low_conf_ratio > 0.60:
            old = config.get("risk", {}).get("min_llm_confidence", 0.3)
            new = max(0.10, round(old - 0.05, 2))
            if new != old and self._allowed("min_llm_confidence"):
                adj["min_llm_confidence"] = new
                rule_triggers.append(
                    f"low_conf_ratio={low_conf_ratio:.2f} > 0.60 → min_llm_confidence {old}→{new}"
                )

        # Rule 3: Evidence blocked — decrease evidence_min_count
        if pending_count > 3 and current_evidence > 1:
            if self._allowed("evidence_min_count"):
                adj["evidence_min_count"] = max(1, current_evidence - 1)
                rule_triggers.append(
                    f"pending_count={pending_count} > 3 → evidence_min_count {current_evidence}→{adj['evidence_min_count']}"
                )

        # Rule 4: Profile churn — disable auto_merge
        if adaptation_count > 3 and current_merge:
            if self._allowed("auto_merge"):
                adj["auto_merge"] = False
                rule_triggers.append(
                    f"adaptation_count={adaptation_count} > 3 → auto_merge False"
                )

        # Rule 5: Recovery — all metrics healthy for 3+ cycles
        all_healthy = (
            overturn_rate <= 0.30
            and low_conf_ratio <= 0.60
            and pending_count <= 3
            and adaptation_count <= 3
        )
        if all_healthy:
            self._healthy_cycles += 1
        else:
            self._healthy_cycles = 0

        if self._healthy_cycles >= 3 and not current_merge:
            if self._allowed("auto_merge"):
                adj["auto_merge"] = True
                rule_triggers.append("3 healthy cycles → auto_merge True")

        if adj and rule_triggers:
            logger.info(
                "ADAPTIVE_FEEDBACK_DETERMINISTIC",
                adjustments=adj,
                triggers=rule_triggers,
                _force_log=True,
            )

        return adj

    # ── LLM reflection (replaces legacy meta-cognition) ─────────────────

    async def _run_llm_reflection(
        self,
        config: dict,
        observations: list[dict],
        beliefs: list[Any],
    ) -> dict:
        if self._llm_registry is None or self._llm_registry.is_degraded():
            logger.info("LLM_REFLECTION_SKIPPED_DEGRADED", reason="registry unavailable or degraded")
            return {}

        interval = config.get("adaptive_feedback", {}).get("llm_interval_cycles", 20)
        if self._cycle_count % interval != 0:
            logger.info("LLM_REFLECTION_SKIPPED_INTERVAL", cycle=self._cycle_count, interval=interval)
            return {}

        metrics = self._compute_recent_metrics()
        safe_config = self._redact_config(config)
        config_str = json.dumps(safe_config, indent=2, default=str)
        research_obs_str = self._get_latest_research_observations()

        win_rate_str = f"{metrics['win_rate']:.1%}" if metrics["win_rate"] is not None else "N/A"
        pf_str = f"{metrics['profit_factor']:.2f}" if metrics["profit_factor"] is not None else "N/A"
        dd_str = f"{metrics['drawdown']:.2%}" if metrics["drawdown"] is not None else "N/A"
        regime_str = "Regime: Unknown / Metrics pending"

        prompt = (
            "You are the Lead Quantitative Researcher for an autonomous crypto trading harness. "
            "You are currently in a reflection cycle. \n\n"
            "Your objective is to analyze recent performance and propose calculated, "
            "evidence-backed adjustments to the system's configuration to progressively "
            "improve its trading logic.\n\n"
            "### RULES OF ENGAGEMENT\n"
            "1. **No Random Tuning:** You must not change parameters just for the sake of "
            "changing them. Every proposed change must be directly justified by the provided "
            "Evidence Observations or Market Structure.\n"
            "2. **Targeted Adjustments:** Do not rewrite the entire configuration. Propose "
            "only the specific parameters that address the identified inefficiencies. \n"
            "3. **Respect the Harness:** You know that the system's research pipeline will "
            "evaluate your changes over the next evaluation cycle. If your changes degrade "
            "performance, the system will automatically revert to the previous state. "
            "Therefore, make conservative, logical adjustments rather than extreme overhauls.\n"
            "4. **Allowed Parameters:** You may only propose changes to Strategy, Research, "
            "Learning, Correlation, and Evidence parameters. You may adjust Execution and "
            "Risk parameters ONLY within their predefined strict bounds. You may NEVER touch "
            "Infrastructure or Guardrail parameters.\n\n"
            "### CURRENT CONTEXT\n"
            f"- **Market Structure:** {regime_str}\n"
            f"- **Recent Performance:** Win Rate: {win_rate_str}, Profit Factor: {pf_str}, "
            f"Drawdown: {dd_str}%\n\n"
            "### EVIDENCE OBSERVATIONS (Generated by Research Pipeline)\n"
            f"{research_obs_str}\n\n"
            "### CURRENT LIVE CONFIGURATION\n"
            f"{config_str}\n\n"
            "### YOUR TASK\n"
            "Analyze the Evidence Observations and Market Structure. Identify the root cause "
            "of any inefficiencies. If a calculated adjustment is warranted, output a strict "
            "JSON payload of the proposed parameter updates.\n\n"
            "### OUTPUT FORMAT\n"
            "Output ONLY a valid JSON object. Do not include markdown formatting, "
            "conversational text, or explanations outside the JSON structure.\n\n"
            "{\n"
            '  "diagnosis": "Brief, precise analysis of what the evidence indicates about '
            "current market structure and strategy performance.\",\n"
            '  "justification": "Why the proposed changes are the most logical, conservative '
            "step to improve the harness, knowing the system will revert if this fails.\",\n"
            '  "parameter_updates": [\n'
            "    {\n"
            '      "key": "research.pattern_overconfidence_high_gap",\n'
            '      "value": 0.25,\n'
            '      "reasoning": "Evidence shows CALIBRATION_DRIFT on SHORTs. Lowering the '
            "gap threshold from 0.3 to 0.25 will make the system more sensitive to "
            "overconfidence.\"\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "If no changes are warranted based on the evidence, return an empty updates array:\n"
            "{\n"
            '  "diagnosis": "...",\n'
            '  "justification": "Current configuration is performing within acceptable '
            "bounds given the market structure. No evidence supports a change at this "
            "cycle.\",\n"
            '  "parameter_updates": []\n'
            "}"
        )

        try:
            messages = [
                {"role": "system", "content": "You are a quantitative researcher. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            raw = await self._llm_registry.chat_completion(
                model=self._llm_model or "",
                messages=messages,
                temperature=0.2,
                max_tokens=500,
            )
        except Exception as e:
            logger.warning("LLM_REFLECTION_FAILED", error=str(e))
            return self._run_deterministic(config, observations)

        try:
            parsed = self._parse_reflection_json(raw)
        except Exception as e:
            logger.warning("LLM_REFLECTION_PARSE_FAILED", error=str(e), raw_preview=raw[:200])
            return self._run_deterministic(config, observations)

        parameter_updates = parsed.get("parameter_updates", [])
        if not parameter_updates:
            logger.info(
                "LLM_REFLECTION_EMPTY",
                diagnosis=parsed.get("diagnosis", "")[:200],
                _force_log=True,
            )
            return {}

        adjustments = self._apply_research_updates(config, parameter_updates)

        if adjustments:
            logger.info(
                "ADAPTIVE_FEEDBACK_LLM_REFLECTION",
                adjustments=adjustments,
                diagnosis=parsed.get("diagnosis", "")[:200],
                _force_log=True,
            )

        return adjustments

    # ── config redaction ────────────────────────────────────────────────

    def _redact_config(self, config: dict) -> dict:
        safe: dict = {}
        for k, v in config.items():
            if k in SENSITIVE_CONFIG_PATHS:
                safe[k] = "<REDACTED>"
            elif isinstance(v, dict):
                safe[k] = self._redact_config(v)
            else:
                safe[k] = v
        return safe

    # ── metrics from evaluation corpus ──────────────────────────────────

    def _compute_recent_metrics(self, lookback_minutes: int = 1440) -> dict[str, Any]:
        result: dict[str, Any] = {
            "win_rate": None,
            "profit_factor": None,
            "drawdown": None,
        }
        if self._eval_corpus is None:
            return result
        try:
            since = datetime.utcnow() - timedelta(minutes=lookback_minutes)
            recent = self._eval_corpus.list_since(since) if hasattr(self._eval_corpus, "list_since") else []
            if not recent:
                return result
            wins = sum(1 for e in recent if e.was_profitable is True)
            losses = sum(1 for e in recent if e.was_profitable is False)
            total = wins + losses
            if total > 0:
                result["win_rate"] = wins / total
            gross_profit = sum(
                e.actual_pnl for e in recent
                if e.was_profitable and e.actual_pnl is not None
            )
            gross_loss = abs(sum(
                e.actual_pnl for e in recent
                if not e.was_profitable and e.actual_pnl is not None
            ))
            if gross_loss > 0:
                result["profit_factor"] = gross_profit / gross_loss
            elif gross_profit > 0:
                result["profit_factor"] = float("inf")
            drawdowns = [e.actual_max_drawdown for e in recent if e.actual_max_drawdown > 0]
            if drawdowns:
                result["drawdown"] = max(drawdowns)
        except Exception:
            logger.warning("METRICS_COMPUTATION_FAILED", exc_info=True)
        return result

    # ── research observations ───────────────────────────────────────────

    def _get_latest_research_observations(self) -> str:
        if self._research_store is None:
            return "No research pipeline connected."
        try:
            reports = self._research_store.list(limit=1)
            if not reports:
                return "No research reports generated yet."
            report = reports[0]
            parts = [
                f"Report {report.report_id} "
                f"(status: {report.status}, sample: {report.sample_size})"
            ]
            for obs in report.observations:
                parts.append(f"- [{obs.category}] {obs.observation}")
            return "\n".join(parts)
        except Exception as e:
            logger.warning("RESEARCH_OBSERVATIONS_FAILED", error=str(e))
            return "Research observations unavailable."

    # ── JSON parsing with markdown stripping ────────────────────────────

    def _parse_reflection_json(self, raw: str) -> dict:
        stripped = raw.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", stripped, re.DOTALL)
        if match:
            stripped = match.group(1).strip()
        if not stripped.startswith("{"):
            brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
            if brace_match:
                stripped = brace_match.group(0)
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError("LLM response is not a dict")
        return parsed

    # ── bounds-checked parameter application ────────────────────────────

    def _apply_research_updates(
        self, config: dict, updates: list[dict],
    ) -> dict[str, Any]:
        applied: dict[str, Any] = {}
        for update in updates:
            key = update.get("key", "")
            value = update.get("value")
            reasoning = update.get("reasoning", "")

            if not key or value is None:
                logger.warning(
                    "LLM_REFLECTION_UPDATE_SKIPPED_MISSING_FIELD",
                    entry=update,
                )
                continue

            bounds = _ALL_BOUNDS.get(key)
            if bounds is None:
                logger.warning(
                    "LLM_REFLECTION_UPDATE_SKIPPED_UNKNOWN_KEY",
                    key=key,
                )
                continue

            lo, hi = bounds
            if lo is not None and hi is not None:
                try:
                    clamped = max(lo, min(hi, float(value)))
                except (TypeError, ValueError):
                    logger.warning(
                        "LLM_REFLECTION_UPDATE_REJECTED_TYPE",
                        key=key,
                        value_type=type(value).__name__,
                    )
                    continue
                if abs(clamped - float(value)) > 1e-9:
                    logger.info(
                        "LLM_REFLECTION_UPDATE_CLAMPED",
                        key=key,
                        original=value,
                        clamped=clamped,
                    )
                final_value = int(clamped) if isinstance(lo, int) and isinstance(hi, int) else clamped
            elif isinstance(value, bool):
                final_value = value
            elif isinstance(value, (int, float)):
                final_value = value
            else:
                logger.warning(
                    "LLM_REFLECTION_UPDATE_REJECTED_TYPE",
                    key=key,
                    value_type=type(value).__name__,
                )
                continue

            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                if part not in target or not isinstance(target[part], dict):
                    target[part] = {}
                target = target[part]
            old = target.get(parts[-1])
            target[parts[-1]] = final_value
            applied[key] = final_value

            logger.info(
                "LLM_REFLECTION_UPDATE_APPLIED",
                key=key,
                old=old,
                new=final_value,
                reasoning=reasoning[:200] if reasoning else "",
            )

        return applied

    # ── helpers ──────────────────────────────────────────────────────────

    def _extract_metrics(self, observations: list[dict]) -> dict:
        metrics: dict[str, Any] = {
            "overturn_rate": 0.0,
            "low_conf_ratio": 0.0,
            "pending_count": 0,
            "adaptation_count": 0,
            "total_trades": 0,
        }
        for obs in observations:
            cat = obs.get("category", "")
            data = obs.get("data", {})
            if cat == "reflection_low_confidence":
                total = data.get("total_actions", 0)
                count = data.get("count", 0)
                metrics["total_trades"] += total
                metrics["low_conf_ratio"] = (count / total) if total > 0 else 0.0
            elif cat == "reflection_critique_overturns":
                total = data.get("total_episodes", 0)
                count = data.get("count", 0)
                metrics["total_trades"] = max(metrics["total_trades"], total)
                metrics["overturn_rate"] = (count / total) if total > 0 else 0.0
            elif cat == "reflection_evidence_pending":
                metrics["pending_count"] += data.get("count", 0)
            elif cat == "reflection_profile_adaptations":
                metrics["adaptation_count"] += data.get("count", 0)
        return metrics

    def _allowed(self, key: str) -> bool:
        remaining = self._cooldowns.get(key, 0)
        if remaining > 0:
            self._cooldowns[key] = remaining - 1
            return False
        return True

    def _apply_cooldowns(self, adjustments: dict) -> None:
        for key in adjustments:
            self._cooldowns[key] = COOLDOWN_CYCLES

    def _persist(
        self, adjustments: dict, config: dict, observations: list[dict],
    ) -> None:
        reasons: list[str] = []
        for obs in observations:
            cat = obs.get("category", "")
            data = obs.get("data", {})
            if cat == "reflection_low_confidence":
                reasons.append(f"low_conf:{data.get('count',0)}/{data.get('total_actions',0)}")
            elif cat == "reflection_critique_overturns":
                reasons.append(f"overturns:{data.get('count',0)}/{data.get('total_episodes',0)}")
            elif cat == "reflection_evidence_pending":
                reasons.append(f"pending:{data.get('count',0)}")
            elif cat == "reflection_profile_adaptations":
                reasons.append(f"adapts:{data.get('count',0)}")

        reason_str = "; ".join(reasons) if reasons else "manual"
        for key, value in adjustments.items():
            str_val = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
            self._config_store.save_feedback_state(key, str_val, reason_str)
