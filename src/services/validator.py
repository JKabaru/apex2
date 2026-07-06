from __future__ import annotations

import math

from src.models.execution import (
    ExecutableTrade,
    ValidationOutcome,
    ValidationOutcomeStatus,
)


class ConsistencyValidator:

    @staticmethod
    def validate(trade: ExecutableTrade) -> list[ValidationOutcome]:
        outcomes: list[ValidationOutcome] = []

        if not trade.symbol:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="EMPTY_SYMBOL",
                message="Symbol must not be empty",
            ))

        if trade.side not in ("BUY", "SELL"):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_SIDE",
                message=f"Side must be BUY or SELL, got {trade.side}",
            ))

        if not (trade.entry_price > 0 and math.isfinite(trade.entry_price)):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_ENTRY_PRICE",
                message=f"Entry price {trade.entry_price} must be positive and finite",
            ))

        if not (trade.quantity > 0 and math.isfinite(trade.quantity)):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_QUANTITY",
                message=f"Quantity {trade.quantity} must be positive and finite",
            ))

        if trade.expected_notional <= 0:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_NOTIONAL",
                message=f"Expected notional {trade.expected_notional} must be positive",
            ))

        if trade.trade_side == "LONG":
            if trade.stop_price > 0 and not (trade.stop_price < trade.entry_price):
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="INVALID_STOP_DIRECTION",
                    message=f"LONG stop {trade.stop_price} must be below entry {trade.entry_price}",
                ))
            if trade.tp_price > 0 and not (trade.tp_price > trade.entry_price):
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="INVALID_TP_DIRECTION",
                    message=f"LONG TP {trade.tp_price} must be above entry {trade.entry_price}",
                ))
        elif trade.trade_side == "SHORT":
            if trade.stop_price > 0 and not (trade.stop_price > trade.entry_price):
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="INVALID_STOP_DIRECTION",
                    message=f"SHORT stop {trade.stop_price} must be above entry {trade.entry_price}",
                ))
            if trade.tp_price > 0 and not (trade.tp_price < trade.entry_price):
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="INVALID_TP_DIRECTION",
                    message=f"SHORT TP {trade.tp_price} must be below entry {trade.entry_price}",
                ))
        else:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_TRADE_SIDE",
                message=f"Trade side must be LONG or SHORT, got {trade.trade_side}",
            ))

        if not trade.execution_id:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="MISSING_EXECUTION_ID",
                message="Execution ID must not be empty",
            ))

        if not trade.trade_group_id:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                code="MISSING_TRADE_GROUP_ID",
                message="Trade group ID is empty",
            ))

        if not (1 <= trade.leverage <= 125):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_LEVERAGE",
                message=f"Leverage {trade.leverage} must be between 1 and 125",
            ))

        if trade.requested_stake <= 0:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_STAKE",
                message=f"Requested stake {trade.requested_stake} must be positive",
            ))

        return outcomes


class IntentValidator:

    @staticmethod
    def validate(trade: ExecutableTrade) -> list[ValidationOutcome]:
        outcomes: list[ValidationOutcome] = []

        expected_position_notional = trade.requested_stake * trade.leverage
        max_allowed_notional = expected_position_notional + trade.notional_tolerance
        if trade.expected_notional > max_allowed_notional:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="NOTIONAL_EXCEEDS_INTENT",
                message=f"Position notional {trade.expected_notional:.4f} exceeds "
                        f"expected {expected_position_notional:.4f} (stake {trade.requested_stake:.4f} × "
                        f"leverage {trade.leverage}) + tolerance {trade.notional_tolerance:.4f}",
            ))
        elif abs(trade.expected_notional - expected_position_notional) > trade.notional_tolerance:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                code="NOTIONAL_DEVIATION",
                message=f"Position notional {trade.expected_notional:.4f} deviates "
                        f"from expected {expected_position_notional:.4f} (stake {trade.requested_stake:.4f} × "
                        f"leverage {trade.leverage}) by "
                        f"{abs(trade.expected_notional - expected_position_notional):.4f} "
                        f"(tolerance {trade.notional_tolerance:.4f})",
            ))

        effective_max_risk = trade.max_allowed_risk + trade.risk_tolerance
        if trade.worst_case_loss > effective_max_risk:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="LOSS_EXCEEDS_RISK_BUDGET",
                message=f"Worst-case loss {trade.worst_case_loss:.6f} exceeds "
                        f"max allowed risk {trade.max_allowed_risk:.6f} + tolerance {trade.risk_tolerance:.6f}",
            ))
        elif trade.worst_case_loss > trade.max_allowed_risk:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                code="LOSS_NEAR_RISK_BUDGET",
                message=f"Worst-case loss {trade.worst_case_loss:.6f} exceeds "
                        f"configured max risk {trade.max_allowed_risk:.6f} but is within tolerance +{trade.risk_tolerance:.6f}",
            ))

        if trade.available_balance > 0 and trade.worst_case_loss > trade.available_balance:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="LOSS_EXCEEDS_BALANCE",
                message=f"Worst-case loss {trade.worst_case_loss:.6f} exceeds available balance {trade.available_balance:.6f}",
            ))

        if trade.expected_reward > 0 and trade.expected_loss > 0:
            rr = trade.expected_reward / trade.expected_loss
            if rr < 1.0:
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                    code="LOW_RISK_REWARD",
                    message=f"Risk/reward ratio {rr:.2f} is below 1.0",
                ))

        return outcomes


class ExchangeValidator:

    @staticmethod
    def validate(trade: ExecutableTrade) -> list[ValidationOutcome]:
        outcomes: list[ValidationOutcome] = []

        if not (trade.quantity > 0 and math.isfinite(trade.quantity)):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_QUANTITY",
                message=f"Quantity {trade.quantity} must be positive and finite",
            ))

        if not (trade.entry_price > 0 and math.isfinite(trade.entry_price)):
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="INVALID_PRICE",
                message=f"Entry price {trade.entry_price} must be positive and finite",
            ))

        if trade.entry_price > 0:
            base_qty = trade.quantity / trade.entry_price

            if trade.step_size > 0:
                step_remainder = base_qty % trade.step_size
                if step_remainder > 1e-12:
                    outcomes.append(ValidationOutcome(
                        status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                        code="STEP_SIZE_ALIGNMENT",
                        message=f"Estimated base qty {base_qty:.6f} (notional {trade.quantity:.4f} / price {trade.entry_price:.4f}) "
                                f"not aligned to step_size {trade.step_size}",
                    ))

            if trade.min_qty > 0 and base_qty < trade.min_qty:
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="QUANTITY_BELOW_MINIMUM",
                    message=f"Estimated base qty {base_qty:.6f} (notional {trade.quantity:.4f} / price {trade.entry_price:.4f}) "
                            f"below min_qty {trade.min_qty}",
                ))

            if trade.max_qty > 0 and base_qty > trade.max_qty:
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.FATAL_FAILURE,
                    code="QUANTITY_ABOVE_MAXIMUM",
                    message=f"Estimated base qty {base_qty:.6f} (notional {trade.quantity:.4f} / price {trade.entry_price:.4f}) "
                            f"above max_qty {trade.max_qty}",
                ))

        if trade.tick_size > 0 and trade.entry_price > 0:
            price_remainder = trade.entry_price % trade.tick_size
            if price_remainder > 1e-12:
                outcomes.append(ValidationOutcome(
                    status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                    code="PRICE_TICK_ALIGNMENT",
                    message=f"Entry price {trade.entry_price} is not aligned to tick_size {trade.tick_size}",
                ))

        effective_min_notional = trade.min_notional if trade.min_notional > 0 else 5.0
        if trade.expected_notional < effective_min_notional:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.FATAL_FAILURE,
                code="NOTIONAL_BELOW_MINIMUM",
                message=f"Position notional {trade.expected_notional:.4f} below minimum "
                        f"{effective_min_notional:.4f}. Configured stake={trade.requested_stake:.2f}, "
                        f"leverage={trade.leverage}x, expected position={trade.requested_stake * trade.leverage:.4f}",
            ))

        if trade.min_notional == 0:
            outcomes.append(ValidationOutcome(
                status=ValidationOutcomeStatus.PASSED_WITH_WARNINGS,
                code="MIN_NOTIONAL_FALLBACK",
                message=f"Exchange did not provide MIN_NOTIONAL filter; using fallback {effective_min_notional}",
            ))

        return outcomes
