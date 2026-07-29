from services.groq import GroqClient
from services.app_logger import get_app_logger

from .agent_state import AgentState


logger = get_app_logger()

class NodeMemoryUpdater:
    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        previous_memory = state.get("convo_memory") or "No previous memory."
        recent_messages = [
            message for message in state["messages"] if message.type in ("human", "ai")
        ][-6:]
        logger.info(f"Recent Messages : {recent_messages}")

        # Fix: Transform the active chat objects into a clean, flat text string
        transcript = ""
        for msg in recent_messages:
            role = "User" if msg.type == "human" else "Assistant"
            transcript += f"{role}: {msg.content}\n"

        logger.info("Memory updater started. Previous memory: %s", previous_memory)
        
        prompt = f"""
            Update the conversation summary using the previous summary and the current
            user/assistant exchange supplied below.

            Previous summary:
            {previous_memory}

            Recent Exchange Transcript:
            {transcript}

            Keep only concise, useful highlights: user preferences, facts they shared,
            decisions made, open tasks, and important context for future replies.
            Do not include tool output, filler, or a transcript. Return only the updated
            summary, with no heading or commentary. Never return an empty response.
        """
        
        # Pass an empty list (or a clean single user message) so Groq sees this as a fresh instruction task
        memory = self.llm.generate_response([], prompt)
        
        if not memory or not memory.strip():
            memory = (
                previous_memory
                if previous_memory != "No previous memory."
                else "No durable memory yet."
            )
            logger.warning("Memory LLM returned an empty response; preserving prior memory.")

        logger.info("Memory updater finished. New memory: %r", memory)
        return {"convo_memory": memory}
