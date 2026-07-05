from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class TradingSession(BaseModel, frozen=True):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    configuration_profile_id: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    git_commit: str = ""
    system_version: str = ""
    operator: str = ""
    startup_reason: str = ""
    hostname: str = ""
    config_hash: str = ""

    @staticmethod
    def compute_config_hash(resolved_configuration: dict[str, Any]) -> str:
        canonical = json.dumps(resolved_configuration, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()
