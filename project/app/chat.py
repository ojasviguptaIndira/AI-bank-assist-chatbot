"""Application service that combines orchestration, response generation, and history."""

from __future__ import annotations

from dataclasses import dataclass, field

from app.chat_history import ChatHistory
from app.conversation import ConversationManager
from app.gemini.response_generator import ResponseGenerator
from app.utils.config import AppConfig
from app.utils.logger import get_logger
from orchestrator.orchestrator import QueryOrchestrator


LOGGER = get_logger(__name__)


@dataclass
class ChatApplication:
    """Coordinate the chat workflow end to end."""

    config: AppConfig = field(default_factory=AppConfig)

    def __post_init__(self) -> None:
        self.history = ChatHistory(limit=self.config.history_limit)
        self.conversation = ConversationManager()
        self.orchestrator = QueryOrchestrator()
        self.response_generator = ResponseGenerator(self.config)

    def process_query(self, query: str) -> dict[str, object]:
        """Resolve context, orchestrate engines, and generate a natural response."""
        resolved_query = self.conversation.resolve_query(query, self.history.list_entries())
        structured = self.orchestrator.handle_query(resolved_query)
        context = self.conversation.build_context(self.history.list_entries())
        generated = self.response_generator.generate(query, structured, context)
        result = {
            "question": query,
            "resolved_query": resolved_query,
            "answer": generated["answer"],
            "structured": structured,
            "classification": structured["classification"],
            "entities": structured["extraction"]["entities"],
            "explainability": generated["explainability"],
            "gemini": generated["gemini"],
        }
        self.history.add(result)
        LOGGER.info(
            "Processed chat query type=%s intent=%s",
            structured["classification"]["type"],
            structured["classification"]["intent"],
        )
        return result

    def clear_history(self) -> None:
        """Clear chat history."""
        self.history.clear()

    def export_history(self) -> str:
        """Export history as JSON."""
        return self.history.export_json()
