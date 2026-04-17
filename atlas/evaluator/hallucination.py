"""Extract atomic claims from generated text.

Breaks a paragraph of generated text into individual factual claims
that can each be independently verified against retrieved evidence.
"""

import json

from openai import OpenAI

from atlas.config import settings
from atlas.observability.logger import get_logger

log = get_logger(__name__)

CLAIM_EXTRACTION_PROMPT = """\
Extract the KEY factual claims from the following text.
Rules:
- Only include concrete, verifiable facts (numbers, names, definitions)
- Skip vague statements like "this is important" or "this is significant"
- Skip meta-statements like "the findings show" or "research indicates"
- Maximum 8 claims
- Each claim must be specific enough to verify against a source

Text:
{text}

Respond with JSON only:
{{"claims": ["claim 1", "claim 2", ...]}}
"""


class ClaimExtractor:
    """Extract atomic factual claims from generated text."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.llm_model
        self._client = OpenAI(api_key=settings.openai_api_key)

    def extract(self, text: str) -> list[str]:
        """Extract atomic claims from a text passage.

        Args:
            text: Generated text to extract claims from.

        Returns:
            List of claim strings.
        """
        if not text or len(text.strip()) < 20:
            return []

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": CLAIM_EXTRACTION_PROMPT.format(text=text),
                    },
                ],
                temperature=0,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            claims = parsed.get("claims", [])

            log.info(
                "claims_extracted",
                num_claims=len(claims),
                text_length=len(text),
            )
            return claims

        except Exception as e:
            log.error("claim_extraction_failed", error=str(e))
            return []
