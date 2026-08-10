from langchain_core.messages import AIMessage

from services.groq import GroqClient
from .agent_state import AgentState
from .node_driver import NodeDriver


class NodeBackup:
    """Recovery-only final answer node when the decision node fails."""

    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        result = NodeDriver._serialize_result(state.get("tool_results"))
        if len(result) > 12_000:
            result = f"Large result schema: {state.get('latest_tool_schema', 'unknown')}"
        prompt = """
            Give the user a useful, concise Markdown answer using the conversation
            and any available tool information. Do not mention internal tools,
            errors, prompts, or routing. If information could not be verified,
            say so plainly and answer from the reliable context available.
        """ + f"\nLatest available tool information:\n{result}"
        try:
            response = self.llm.generate_response(state["messages"], prompt)
        except Exception:
            response = "I’m sorry, but I couldn’t complete that request reliably."
        return {
            "messages": [AIMessage(content=response)],
            "final_response": [response],
            "status_messages": ["complete"],
        }
