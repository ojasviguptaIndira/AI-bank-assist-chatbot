"""Generate natural-language answers from structured orchestrator output."""

from __future__ import annotations

import time
from dataclasses import dataclass

from app.gemini.client import GeminiClient
from app.gemini.prompt_builder import PromptBuilder
from app.gemini.response_parser import ResponseParser
from app.utils.config import AppConfig
from app.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass
class ResponseGenerator:
    """Convert structured JSON into conversational responses."""

    config: AppConfig

    def __post_init__(self) -> None:
        self.client = GeminiClient(self.config)
        self.prompt_builder = PromptBuilder()
        self.parser = ResponseParser()

    def generate(
        self,
        user_query: str,
        structured_payload: dict[str, object],
        conversation_context: str = "",
    ) -> dict[str, object]:
        """Generate a natural-language response with explainability metadata."""
        started = time.perf_counter()
        prompt = self.prompt_builder.build(user_query, structured_payload, conversation_context)
        gemini_result = self.client.generate(prompt)
        if gemini_result["success"] and gemini_result["text"].strip():
            answer = self.parser.parse(gemini_result["text"])
        else:
            answer = self._fallback_answer(structured_payload)

        explainability = self._build_explainability(structured_payload)
        return {
            "answer": answer,
            "explainability": explainability,
            "gemini": {
                "available": self.client.available,
                "model": gemini_result["model"],
                "status": "success" if gemini_result["success"] else "fallback",
                "error": gemini_result["error"],
                "execution_time_sec": round(time.perf_counter() - started, 4),
            },
            "prompt": prompt,
        }

    def _fallback_answer(self, structured_payload: dict[str, object]) -> str:
        response = structured_payload.get("response", {})
        response_type = response.get("type")
        if response_type == "analytics":
            data = response.get("data", [])
            if not data:
                return "I could not find enough analytical data to answer that query."
            return f"Here is the analytical result based on the historical loan dataset: {data[0]}."
        if response_type == "sentiment":
            prediction = response.get("prediction", "Unknown")
            confidence = response.get("confidence")
            bank = response.get("bank", "the selected bank")
            if confidence is not None:
                return f"Based on the analyzed customer reviews, {bank} appears {prediction.lower()} with confidence of {confidence:.0%}."
            return f"Based on the analyzed customer reviews, the predicted sentiment is {prediction.lower()}."
        if response_type == "hybrid":
            return (
                "I combined the available analytics and sentiment outputs. "
                "Bank-level sentiment comparison is available, but bank-level loan pricing is limited in the current dataset."
            )
        return response.get("message", "I could not determine a supported answer for that query.")

    def _build_explainability(self, structured_payload: dict[str, object]) -> dict[str, object]:
        response = structured_payload.get("response", {})
        response_type = response.get("type", "unknown")
        timing = structured_payload.get("timing", {})
        metadata = response.get("metadata", {})
        source = metadata.get("source", "Unknown")
        execution_time = timing.get("total_time_sec")
        analytics_time = None
        sentiment_time = None
        if response_type == "analytics":
            analytics_time = metadata.get("execution_time_sec")
        elif response_type == "sentiment":
            sentiment_time = metadata.get("execution_time_sec")
        elif response_type == "hybrid":
            sentiment_time = response.get("metadata", {}).get("execution_time_sec")
        return {
            "source": source,
            "records_analyzed": metadata.get("records_analyzed"),
            "confidence": response.get("confidence"),
            "generated_using": metadata.get("generated_using", "Gemini Response Generator"),
            "sql_executed": metadata.get("sql_executed"),
            "analytics_time_sec": analytics_time,
            "sentiment_time_sec": sentiment_time,
            "total_time_sec": execution_time,
        }
