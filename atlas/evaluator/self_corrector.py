"""Closed-Loop Self-Correcting RAG with Grounding-Guided Re-Retrieval.

The novel contribution: after generating an answer, the system:
1. Extracts claims and scores groundedness (existing evaluator)
2. Identifies unsupported claims
3. Generates targeted re-retrieval queries for unsupported claims
4. Re-retrieves evidence specifically for weak claims
5. Regenerates ONLY the unsupported sections with new evidence
6. Re-evaluates to measure improvement
7. Repeats until convergence or max cycles

This creates a closed feedback loop that existing RAG systems lack.
"""

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI

from atlas.config import settings
from atlas.evaluator.confidence import EvaluationReport
from atlas.evaluator.evaluator import HallucinationEvaluator
from atlas.observability.logger import get_logger
from atlas.observability.tracer import StepType, Trace, TraceStep
from atlas.retriever.arxiv_search import ArxivSearchTool
from atlas.retriever.web_search import WebSearchTool

log = get_logger(__name__)

REQUERY_PROMPT = """\
You are given a list of factual claims that were NOT supported by \
the retrieved evidence. For each unsupported claim, generate a \
specific search query that would find evidence to verify or refute it.

Unsupported claims:
{claims}

Respond with JSON only:
{{"queries": ["search query 1", "search query 2", ...]}}
"""

REGENERATE_PROMPT = """\
You previously generated an answer to a research question, but some \
claims were not supported by evidence. You now have NEW evidence \
for those claims. Rewrite ONLY the parts of the answer that were \
unsupported, keeping the supported parts unchanged.

Original answer:
{original_answer}

Unsupported claims that need correction:
{unsupported_claims}

New evidence found for those claims:
{new_evidence}

Rewrite the full answer, correcting or removing unsupported claims \
based on the new evidence. If the new evidence contradicts a claim, \
update it. If no evidence was found, note the claim as unverified.
"""


@dataclass
class CorrectionCycle:
    """Record of one correction cycle."""

    cycle_number: int
    claims_before: int = 0
    supported_before: int = 0
    unsupported_before: int = 0
    confidence_before: float = 0.0
    requery_count: int = 0
    new_evidence_count: int = 0
    claims_after: int = 0
    supported_after: int = 0
    confidence_after: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle_number,
            "before": {
                "claims": self.claims_before,
                "supported": self.supported_before,
                "unsupported": self.unsupported_before,
                "confidence": round(self.confidence_before, 4),
            },
            "re_retrieval": {
                "queries": self.requery_count,
                "new_evidence": self.new_evidence_count,
            },
            "after": {
                "claims": self.claims_after,
                "supported": self.supported_after,
                "confidence": round(self.confidence_after, 4),
            },
            "improvement": round(
                self.confidence_after - self.confidence_before, 4
            ),
            "latency_ms": round(self.latency_ms, 2),
        }


@dataclass
class SelfCorrectionResult:
    """Full result of the self-correction process."""

    original_answer: str
    corrected_answer: str
    original_confidence: float
    final_confidence: float
    num_cycles: int
    cycles: list[CorrectionCycle] = field(default_factory=list)
    original_evaluation: EvaluationReport | None = None
    final_evaluation: EvaluationReport | None = None
    converged: bool = False

    def to_dict(self) -> dict:
        return {
            "original_confidence": round(self.original_confidence, 4),
            "final_confidence": round(self.final_confidence, 4),
            "improvement": round(
                self.final_confidence - self.original_confidence, 4
            ),
            "num_cycles": self.num_cycles,
            "converged": self.converged,
            "cycles": [c.to_dict() for c in self.cycles],
        }


class SelfCorrector:
    """Closed-loop self-correcting RAG system.

    Takes a generated answer + evidence, evaluates it, then iteratively
    re-retrieves and regenerates to improve groundedness.
    """

    def __init__(
        self,
        max_cycles: int = 3,
        convergence_threshold: float = 0.05,
        model: str | None = None,
    ):
        """
        Args:
            max_cycles: Maximum correction cycles before stopping.
            convergence_threshold: Stop if improvement < this value.
            model: LLM model for re-query generation and regeneration.
        """
        self.max_cycles = max_cycles
        self.convergence_threshold = convergence_threshold
        self.model = model or settings.llm_model
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._evaluator = HallucinationEvaluator()
        self._web_search = WebSearchTool()
        self._arxiv_search = ArxivSearchTool()

        log.info(
            "self_corrector_ready",
            max_cycles=max_cycles,
            threshold=convergence_threshold,
        )

    def correct(
        self,
        answer: str,
        evidence_chunks: list[dict],
        trace: Trace | None = None,
        on_cycle: callable = None,
    ) -> SelfCorrectionResult:
        """Run the self-correction loop.

        Args:
            answer: The original generated answer.
            evidence_chunks: Evidence from the initial retrieval.
            trace: Optional trace for observability.
            on_cycle: Optional callback called after each cycle
                      with (cycle_number, CorrectionCycle).

        Returns:
            SelfCorrectionResult with correction history.
        """
        current_answer = answer
        current_evidence = list(evidence_chunks)
        cycles = []

        # Initial evaluation
        initial_eval = self._evaluator.evaluate(
            current_answer, current_evidence
        )
        original_confidence = initial_eval.overall_confidence

        log.info(
            "self_correction_started",
            original_confidence=original_confidence,
            total_claims=initial_eval.total_claims,
        )

        prev_confidence = original_confidence

        for cycle_num in range(1, self.max_cycles + 1):
            cycle_start = time.time()

            # Get unsupported claims
            unsupported = [
                cr for cr in initial_eval.claim_results
                if cr["status"] != "supported"
            ]

            if not unsupported:
                log.info(
                    "all_claims_supported",
                    cycle=cycle_num,
                )
                cycle = CorrectionCycle(
                    cycle_number=cycle_num,
                    claims_before=initial_eval.total_claims,
                    supported_before=initial_eval.supported_claims,
                    confidence_before=prev_confidence,
                    claims_after=initial_eval.total_claims,
                    supported_after=initial_eval.supported_claims,
                    confidence_after=prev_confidence,
                )
                cycles.append(cycle)
                break

            log.info(
                "correction_cycle",
                cycle=cycle_num,
                unsupported_claims=len(unsupported),
            )

            # Step 1: Generate targeted re-retrieval queries
            unsupported_texts = [c["claim"] for c in unsupported]
            queries = self._generate_requery(unsupported_texts)

            # Step 2: Re-retrieve with targeted queries
            new_evidence = self._re_retrieve(queries)

            # Step 3: Regenerate answer with new evidence
            current_answer = self._regenerate(
                current_answer, unsupported_texts, new_evidence
            )

            # Add new evidence to pool
            current_evidence.extend(new_evidence)

            # Step 4: Re-evaluate
            new_eval = self._evaluator.evaluate(
                current_answer, current_evidence
            )

            cycle_end = time.time()

            cycle = CorrectionCycle(
                cycle_number=cycle_num,
                claims_before=initial_eval.total_claims,
                supported_before=initial_eval.supported_claims,
                unsupported_before=len(unsupported),
                confidence_before=prev_confidence,
                requery_count=len(queries),
                new_evidence_count=len(new_evidence),
                claims_after=new_eval.total_claims,
                supported_after=new_eval.supported_claims,
                confidence_after=new_eval.overall_confidence,
                latency_ms=(cycle_end - cycle_start) * 1000,
            )
            cycles.append(cycle)

            if on_cycle:
                on_cycle(cycle_num, cycle)

            log.info(
                "cycle_complete",
                cycle=cycle_num,
                confidence_before=round(prev_confidence, 4),
                confidence_after=round(new_eval.overall_confidence, 4),
                improvement=round(
                    new_eval.overall_confidence - prev_confidence, 4
                ),
            )

            # Add to trace
            if trace:
                trace.add_step(
                    TraceStep(
                        step_id=f"correction_cycle_{cycle_num}",
                        step_type=StepType.EVALUATION,
                        input_data={
                            "unsupported_claims": len(unsupported),
                            "requery_count": len(queries),
                        },
                        output_data=cycle.to_dict(),
                        start_time=cycle_start,
                        end_time=cycle_end,
                    )
                )

            # Check convergence
            improvement = new_eval.overall_confidence - prev_confidence
            if improvement < self.convergence_threshold:
                log.info(
                    "convergence_reached",
                    improvement=round(improvement, 4),
                    threshold=self.convergence_threshold,
                )
                initial_eval = new_eval
                prev_confidence = new_eval.overall_confidence
                break

            initial_eval = new_eval
            prev_confidence = new_eval.overall_confidence

        converged = (
            prev_confidence >= 0.9
            or len(cycles) < self.max_cycles
        )

        result = SelfCorrectionResult(
            original_answer=answer,
            corrected_answer=current_answer,
            original_confidence=original_confidence,
            final_confidence=prev_confidence,
            num_cycles=len(cycles),
            cycles=cycles,
            original_evaluation=None,
            final_evaluation=initial_eval,
            converged=converged,
        )

        log.info(
            "self_correction_complete",
            original=round(original_confidence, 4),
            final=round(prev_confidence, 4),
            cycles=len(cycles),
            converged=converged,
        )

        return result

    def _generate_requery(self, unsupported_claims: list[str]) -> list[str]:
        """Generate targeted search queries for unsupported claims."""
        try:
            claims_text = "\n".join(
                f"- {c}" for c in unsupported_claims[:5]
            )
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": REQUERY_PROMPT.format(
                            claims=claims_text
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            queries = parsed.get("queries", [])

            log.info("requery_generated", num_queries=len(queries))
            return queries[:5]

        except Exception as e:
            log.error("requery_failed", error=str(e))
            return [c[:100] for c in unsupported_claims[:3]]

    def _re_retrieve(self, queries: list[str]) -> list[dict]:
        """Re-retrieve evidence using targeted queries."""
        new_evidence = []

        for query in queries:
            # Try web search
            if self._web_search.is_available:
                results = self._web_search.search(
                    query, max_results=2
                )
                for r in results:
                    new_evidence.append({
                        "text": r.get("content", "")[:800],
                        "source": f"re-retrieval:web:{query[:50]}",
                    })

            # Try arxiv
            results = self._arxiv_search.search(
                query, max_results=1
            )
            for r in results:
                new_evidence.append({
                    "text": r.get("abstract", "")[:800],
                    "source": f"re-retrieval:arxiv:{r.get('arxiv_id', '')}",
                })

        log.info(
            "re_retrieval_complete",
            queries=len(queries),
            new_evidence=len(new_evidence),
        )
        return new_evidence

    def _regenerate(
        self,
        original_answer: str,
        unsupported_claims: list[str],
        new_evidence: list[dict],
    ) -> str:
        """Regenerate the answer using new evidence."""
        try:
            claims_text = "\n".join(
                f"- {c}" for c in unsupported_claims
            )
            evidence_text = "\n\n".join(
                f"[{e['source']}]: {e['text'][:400]}"
                for e in new_evidence
            )

            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": REGENERATE_PROMPT.format(
                            original_answer=original_answer[:2000],
                            unsupported_claims=claims_text,
                            new_evidence=evidence_text[:3000],
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=settings.llm_max_tokens,
            )

            regenerated = response.choices[0].message.content
            log.info(
                "answer_regenerated",
                original_len=len(original_answer),
                new_len=len(regenerated or ""),
            )
            return regenerated or original_answer

        except Exception as e:
            log.error("regeneration_failed", error=str(e))
            return original_answer
