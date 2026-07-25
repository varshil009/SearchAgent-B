from services.groq import GroqClient
from services.app_logger import get_app_logger

from .agent_state import AgentState


logger = get_app_logger()


class NodeMemoryUpdater:
    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        previous_memory = state.get("convo_memory") or "No previous memory."
        latest_human_message = next(
            (message for message in reversed(state["messages"]) if message.type == "human"),
            None,
        )
        latest_ai_message = next(
            (message for message in reversed(state["messages"]) if message.type == "ai"),
            None,
        )
        recent_messages = [
            message for message in (latest_human_message, latest_ai_message) if message
        ]

        logger.info("Memory updater started. Previous memory: %s", previous_memory)
        prompt = f"""
            Update the conversation using the previous summary and the current
            user/assistant exchange supplied below.

            Previous summary:
            {previous_memory}

            Keep only concise, useful highlights: user preferences, facts they shared,
            decisions made, open tasks, and important context for future replies.
            Do not include tool output, filler, or a transcript. Return only the updated
            summary, with no heading or commentary. Never return an empty response.
        """
        memory = self.llm.generate_response(recent_messages, prompt)
        if not memory or not memory.strip():
            memory = (
                previous_memory
                if previous_memory != "No previous memory."
                else "No durable memory yet."
            )
            logger.warning("Memory LLM returned an empty response; preserving prior memory.")

        logger.info("Memory updater finished. New memory: %r", memory)
        return {"convo_memory": memory}
