from __future__ import annotations

import structlog

from src.evaluation.storage import EvaluationCorpus
from src.research.analyzer import ResearchAnalyzer
from src.research.storage import ResearchCorpusStore

logger = structlog.get_logger("research_pipeline")


class ResearchPipeline:
    """Orchestrates research report generation from the evaluation corpus.

    Not triggered automatically — operator-initiated on demand.
    """

    def __init__(
        self,
        evaluation_corpus: EvaluationCorpus,
        research_store: ResearchCorpusStore,
    ):
        self._eval_corpus = evaluation_corpus
        self._research_store = research_store

    def generate_research_report(self, version: str = "1.0") -> str:
        """Load all evaluations, analyze, persist report, return report_id."""
        evaluations = self._eval_corpus.list(limit=0)
        logger.info(
            "Loaded evaluations for research",
            count=len(evaluations),
            version=version,
        )

        report = ResearchAnalyzer.analyze(evaluations, version=version)
        self._research_store.save(report)

        logger.info(
            "Research report generated",
            report_id=report.report_id,
            status=report.status,
            sample_size=report.sample_size,
        )
        return report.report_id
