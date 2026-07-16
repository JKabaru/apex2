from __future__ import annotations

import json
from typing import Any, Optional

import structlog

from src.llm.registry import LLMRegistry

logger = structlog.get_logger("adaptive_feedback")

DEFAULT_CONFIG = {
    "auto_merge": True,
    "show_confidence_in_prompt": True,
    "evidence_min_count": 1,
}

COOLDOWN_CYCLES = 2


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
    ):
        self._config_store = config_store
        self._llm_registry = llm_registry
        self._llm_model = llm_model
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
            llm_adj = await self._run_llm_meta(config, observations, beliefs)
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

    # ── LLM meta-cognition ──────────────────────────────────────────────

    async def _run_llm_meta(
        self,
        config: dict,
        observations: list[dict],
        beliefs: list[Any],
    ) -> dict:
        if self._llm_registry is None or self._llm_registry.is_degraded():
            logger.info("LLM meta-cognition skipped (registry unavailable or degraded)")
            return {}

        interval = config.get("adaptive_feedback", {}).get("llm_interval_cycles", 20)
        if self._cycle_count % interval != 0:
            return {}

        metrics = self._extract_metrics(observations)
        current_merge = config.get("adaptive", {}).get("auto_merge", True)
        current_confidence = config.get("adaptive", {}).get("show_confidence_in_prompt", True)
        current_evidence = config.get("learning", {}).get("evidence_min_count", 1)
        current_min_conf = config.get("risk", {}).get("min_llm_confidence", 0.3)

        prompt = (
            "You are a trading system optimizer. Adjust 4 settings based on recent metrics.\n\n"
            f"Metrics:\n"
            f"- Recent trades: {metrics.get('total_trades', 0)}\n"
            f"- Critique overturn rate: {metrics.get('overturn_rate', 0.0):.2f}\n"
            f"- Low-confidence action ratio: {metrics.get('low_conf_ratio', 0.0):.2f}\n"
            f"- Pending evidence items: {metrics.get('pending_count', 0)}\n"
            f"- Profile adaptations: {metrics.get('adaptation_count', 0)}\n\n"
            "Current values:\n"
            f"- auto_merge: {current_merge}\n"
            f"- show_confidence_in_prompt: {current_confidence}\n"
            f"- evidence_min_count: {current_evidence}\n"
            f"- min_llm_confidence: {current_min_conf}\n\n"
            "Output ONLY valid JSON with keys you want to change:\n"
            '{"auto_merge": bool, "show_confidence_in_prompt": bool, '
            '"evidence_min_count": int (>=1), "min_llm_confidence": float (0.1-0.8)}\n'
            "Omit keys you want to keep unchanged."
        )

        try:
            messages = [
                {"role": "system", "content": "You are a trading system optimizer. Output only valid JSON."},
                {"role": "user", "content": prompt},
            ]
            raw = await self._llm_registry.chat_completion(
                model=self._llm_model or "",
                messages=messages,
                temperature=0.2,
                max_tokens=150,
            )
        except Exception as e:
            logger.warning("LLM meta-cognition failed, falling back to deterministic", error=str(e))
            return self._run_deterministic(config, observations)

        try:
            parsed = json.loads(raw.strip())
            if not isinstance(parsed, dict):
                raise ValueError("LLM response is not a dict")
            adj: dict = {}
            for key in ("auto_merge", "show_confidence_in_prompt"):
                if key in parsed and isinstance(parsed[key], bool):
                    adj[key] = parsed[key]
            if "evidence_min_count" in parsed:
                adj["evidence_min_count"] = max(1, int(parsed["evidence_min_count"]))
            if "min_llm_confidence" in parsed:
                adj["min_llm_confidence"] = round(max(0.1, min(0.8, float(parsed["min_llm_confidence"]))), 2)

            if adj:
                logger.info("ADAPTIVE_FEEDBACK_LLM", adjustments=adj, _force_log=True)

            return adj
        except Exception as e:
            logger.warning("LLM meta-cognition parse failed, falling back", error=str(e))
            return self._run_deterministic(config, observations)

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
