"""NLI-based grounding verification.

Uses a Natural Language Inference model to score whether each claim
is ENTAILED (supported), CONTRADICTED, or NEUTRAL (not enough info)
by the retrieved evidence chunks.

Model: facebook/bart-large-mnli
Uses the model directly for NLI (premise=evidence, hypothesis=claim).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from atlas.observability.logger import get_logger

log = get_logger(__name__)

# Label mapping for BART-MNLI: index 0=contradiction, 1=neutral, 2=entailment
LABEL_MAP = {0: "contradiction", 1: "neutral", 2: "entailment"}


@dataclass
class GroundingResult:
    """Result of grounding a single claim against evidence."""

    claim: str
    label: str  # "entailment", "contradiction", or "neutral"
    score: float  # Confidence in the label (0.0 - 1.0)
    evidence: str  # The evidence chunk that was checked
    evidence_source: str  # Source of the evidence (filename, URL, etc.)

    @property
    def is_supported(self) -> bool:
        return self.label == "entailment" and self.score >= 0.5

    @property
    def is_contradicted(self) -> bool:
        return self.label == "contradiction" and self.score >= 0.5

    def to_dict(self) -> dict:
        return {
            "claim": self.claim,
            "label": self.label,
            "score": round(self.score, 4),
            "is_supported": self.is_supported,
            "evidence_preview": self.evidence[:200],
            "evidence_source": self.evidence_source,
        }


class GroundingScorer:
    """Score claims against evidence using NLI entailment.

    For each (claim, evidence_chunk) pair, the NLI model produces
    probabilities for:
    - entailment: evidence supports the claim
    - contradiction: evidence contradicts the claim
    - neutral: evidence doesn't address the claim

    We check each claim against ALL evidence chunks and take
    the best entailment score.
    """

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        log.info("loading_nli_model", model=model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self._model.eval()
        log.info("nli_model_ready", model=model_name)

    def _score_pair(self, premise: str, hypothesis: str) -> dict:
        """Score a single (premise, hypothesis) pair with NLI.

        Args:
            premise: The evidence text.
            hypothesis: The claim to verify.

        Returns:
            Dict with entailment, contradiction, neutral scores.
        """
        inputs = self._tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1)[0]

        return {
            "contradiction": float(probs[0]),
            "neutral": float(probs[1]),
            "entailment": float(probs[2]),
        }

    def score_claim(
        self,
        claim: str,
        evidence_chunks: list[dict],
    ) -> GroundingResult:
        """Score a single claim against multiple evidence chunks.

        Checks the claim against each chunk and returns the result
        with the highest entailment score.

        Args:
            claim: The factual claim to verify.
            evidence_chunks: List of dicts with 'text' and 'source' keys.

        Returns:
            GroundingResult with the best evidence match.
        """
        if not evidence_chunks:
            return GroundingResult(
                claim=claim,
                label="neutral",
                score=0.0,
                evidence="",
                evidence_source="no evidence",
            )

        best_result = None
        # Track by (entailment_is_dominant, entailment_score) so chunks
        # where entailment IS the top label are always preferred over
        # chunks where contradiction/neutral dominates.
        best_key = (False, -1.0)

        for chunk in evidence_chunks:
            text = chunk.get("text", "")
            source = chunk.get("source", "unknown")

            if not text.strip():
                continue

            # Truncate long evidence
            truncated = text[:512]

            try:
                scores = self._score_pair(
                    premise=truncated, hypothesis=claim
                )

                entailment_score = scores["entailment"]
                top_label = max(scores, key=scores.get)
                top_score = scores[top_label]

                # Prefer chunks where entailment is the dominant label,
                # then by entailment score as tiebreaker
                key = (top_label == "entailment", entailment_score)

                if key > best_key:
                    best_key = key

                    best_result = GroundingResult(
                        claim=claim,
                        label=top_label,
                        score=round(top_score, 4),
                        evidence=text,
                        evidence_source=source,
                    )

            except Exception as e:
                log.error(
                    "nli_scoring_failed",
                    claim=claim[:60],
                    error=str(e),
                )
                continue

        if best_result is None:
            return GroundingResult(
                claim=claim,
                label="neutral",
                score=0.0,
                evidence="",
                evidence_source="scoring failed",
            )

        return best_result

    def score_claims(
        self,
        claims: list[str],
        evidence_chunks: list[dict],
    ) -> list[GroundingResult]:
        """Score multiple claims against evidence.

        Args:
            claims: List of factual claims.
            evidence_chunks: List of dicts with 'text' and 'source' keys.

        Returns:
            List of GroundingResult, one per claim.
        """
        results = []
        for claim in claims:
            result = self.score_claim(claim, evidence_chunks)
            results.append(result)

            log.info(
                "claim_scored",
                claim=claim[:60],
                label=result.label,
                score=result.score,
                supported=result.is_supported,
            )

        return results
