from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from src.models.tim.enums import TIMMode


class TIMConfig(BaseModel):
    tim_mode: TIMMode = TIMMode.OFF
    watchdog_timeout_minutes: int = 60
    max_intent_retries: int = 3
    default_review_interval_minutes: int = 240
    max_journal_entries_before_compression: int = 500
    prompt_version: str = "1.0"
    schema_version: str = "1.0"
    config_version: str = "1.0"

    @model_validator(mode="after")
    def _validate_mode(self) -> TIMConfig:
        if not isinstance(self.tim_mode, TIMMode):
            raise ValueError(f"Invalid tim_mode: {self.tim_mode}")
        return self
