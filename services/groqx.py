import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq, APIError, APITimeoutError, RateLimitError

from services.app_logger import get_app_logger

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

logger = get_app_logger("groqx")


class GroqClient:
    def __init__(self):
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=30.0,
        )
        self.active_model = os.getenv("GROQ_MODEL")
        self.models = [
            "qwen/qwen3.6-27b",
            "groq/compound",
            "groq/compound-mini",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "openai/gpt-oss-safeguard-20b",
        ]
        self.max_retries = 3
        self.request_timeout = 60.0

    def _find_model_index(self) -> int:
        """Return the index of *active_model* in *self.models*, or 0."""
        try:
            return self.models.index(self.active_model)
        except ValueError:
            logger.warning("active_model %r not found in models list, defaulting to index 0", self.active_model)
            return 0

    def _switch_to_next_model(self):
        """Advance *active_model* to the next model in the list (round-robin)."""
        idx = self._find_model_index()
        next_idx = (idx + 1) % len(self.models)
        self.active_model = self.models[next_idx]
        logger.info("Switched active model to %r (index %d)", self.active_model, next_idx)

    def generate_response(self, convo, system_prompt):
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        ## messages are created in groq supported formate;
        ## [{"role" : "user", "content" : "asdfghjk..."}, 
        ##  {"role" : "assistant", "content" : "asdfghjk..."]
        for message in convo:
            role = "user" if message.type == "human" else "assistant"

            messages.append({
                "role": role,
                "content": message.content
            })

        retries = 0
        failed_models_count = 0
        while failed_models_count < len(self.models):
            try:
                response = self.client.chat.completions.create(
                    model=self.active_model,
                    messages=messages,
                    timeout=self.request_timeout,
                )
                logger.info("Groq API success with model %r", self.active_model)
                return response.choices[0].message.content

            except RateLimitError as exc:
                # Groq includes a 'retry-after' header value inside the exception context if available
                # Fall back to exponential if the header is missing
                sleep_sec = getattr(exc, 'retry_after', 2 ** (retries + 1))
                retries += 1
                
                logger.warning("Rate Limit on %r. Attempt %d/%d. Waiting %ds...", self.active_model, retries, self.max_retries, sleep_sec)
                
                if retries >= self.max_retries:
                    logger.error("Model %r rate limits exhausted. Cycling model.", self.active_model)
                    self._switch_to_next_model()
                    failed_models_count += 1
                    retries = 0
                    time.sleep(1.0)
                else:
                    time.sleep(sleep_sec)

            except (APITimeoutError, APIError) as exc:
                # Check for bad API Key or structural formatting errors (HTTP 401, 400)
                # These are unrecoverable; do not waste time retrying them
                if hasattr(exc, 'status_code') and exc.status_code in (400, 401):
                    logger.error("Unrecoverable API error (%d): %s. Halting.", exc.status_code, exc)
                    raise exc

                retries += 1
                sleep_sec = 2 ** retries
                logger.warning("API Network Error (%s) on %r. Retrying in %ds...", type(exc).__name__, self.active_model, sleep_sec)
                
                if retries >= self.max_retries:
                    logger.error("Model %r exhausted retries due to errors. Cycling model.", self.active_model)
                    self._switch_to_next_model()
                    failed_models_count += 1
                    retries = 0
                    time.sleep(1.0)
                else:
                    time.sleep(sleep_sec)
                    
        # If the execution breaks out of the loop, every single model in your array failed
        critical_error_msg = "All available models in the harness failed to produce a response."
        logger.critical(critical_error_msg)
        raise RuntimeError(critical_error_msg)