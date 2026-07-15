"""Gemini client wrapper."""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty

from app.utils.config import AppConfig
from app.utils.logger import get_logger


LOGGER = get_logger(__name__)

GEMINI_TIMEOUT_SEC = 12


def _generate_with_google_api(
    api_key: str,
    model_name: str,
    prompt: str,
    output_queue: mp.Queue,
) -> None:
    """Execute Gemini generation in a child process."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        output_queue.put(
            {
                "success": True,
                "text": getattr(response, "text", "") or "",
                "model": model_name,
                "error": "",
            }
        )
    except Exception as exc:  # pragma: no cover - child-process fallback
        output_queue.put(
            {
                "success": False,
                "text": "",
                "model": model_name,
                "error": str(exc),
            }
        )


@dataclass
class GeminiClient:
    """Thin wrapper around the Gemini text generation API."""

    config: AppConfig

    def __post_init__(self) -> None:
        self.available = False
        self.error_message = ""
        try:
            import google.generativeai as genai
        except ImportError:
            self.error_message = "google-generativeai is not installed."
            self._genai = None
            return

        if not self.config.gemini_ready:
            self.error_message = "Gemini API key is not configured."
            self._genai = genai
            return

        self._genai = genai
        self._genai.configure(api_key=self.config.gemini_api_key)
        self.available = True

    def generate(self, prompt: str) -> dict[str, object]:
        """Generate a response from Gemini."""
        if not self.available or self._genai is None:
            return {
                "success": False,
                "text": "",
                "model": self.config.gemini_model,
                "error": self.error_message or "Gemini is unavailable.",
            }

        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        output_queue: mp.Queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_generate_with_google_api,
            args=(self.config.gemini_api_key, self.config.gemini_model, prompt, output_queue),
            daemon=True,
        )
        process.start()
        process.join(GEMINI_TIMEOUT_SEC)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            return {
                "success": False,
                "text": "",
                "model": self.config.gemini_model,
                "error": f"Gemini request timed out after {GEMINI_TIMEOUT_SEC} seconds.",
            }

        try:
            result = output_queue.get_nowait()
        except Empty:
            result = {
                "success": False,
                "text": "",
                "model": self.config.gemini_model,
                "error": "Gemini did not return a response.",
            }
        finally:
            output_queue.close()

        if not result["success"]:
            LOGGER.warning("Gemini generation failed: %s", result["error"])
        return result
