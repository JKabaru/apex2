from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class CorpusMetadata(BaseModel, frozen=True):
    """Fingerprint of the corpus itself. Stored once, referenced by every manifest.
    If two corpora have the same fingerprint, they are structurally compatible."""
    corpus_schema_version: str = "1.0"
    pipeline_version: str = "1.0"
    feature_catalog_hash: str = ""
    config_catalog_version: str = ""
    provenance_version: str = ""
    application_version: str = ""
    git_commit: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
