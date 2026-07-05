from __future__ import annotations

from src.retrieval.models import CorpusDiagnostics
from src.storage.learning.learning_corpus import LearningCorpus


class CorpusHealthAnalyzer:
    """Diagnostic tool for corpus quality assessment.
    Measures corpus quality — not learning quality.
    Never modifies the corpus. Read-only analysis."""

    def __init__(self, corpus: LearningCorpus):
        self._corpus = corpus

    def analyze(self) -> CorpusDiagnostics:
        return self._corpus.get_diagnostics()
