from langchain_core.messages import AIMessage

from services.groq import GroqClient
from services.app_logger import get_app_logger
from .agent_state import AgentState
from .node_driver import NodeDriver


logger = get_app_logger("node_llm")


class NodeLLM:
    """Initial decision node for a graph invocation.

    NodeLLM decides whether the user's request can be answered immediately or
    needs its first tool call. Once a tool returns, NodeDriver owns all subsequent
    decisions in the cyclic portion of the graph.
    """

    def __init__(self):
        self.llm = GroqClient()
        # The first decision follows the identical strict output contract as the
        # post-tool decision node, but has no previous tool result to consider.
        self.decision_prompt = NodeDriver()._prompt(
            {
                "tool_results": None,
                "latest_tool_schema": "No tool result is available yet.",
            },
            loop_blocked=False,
        )

    def generate(self, state: AgentState):
        prompt = f"""
            {self.decision_prompt}

            Conversation memory from earlier API calls:
            {state.get("convo_memory") or "No saved memory yet."}
        """
        try:
            response = self.llm.generate_response(state["messages"], prompt)
        except Exception:
            logger.exception("NodeLLM request failed")
            return {"next_action": "backup", "status_messages": ["generating"]}

        logger.info("NodeLLM raw LLM response (run=%s): %r", state.get("tool_run_id"), response)

        final_text = NodeDriver._extract_final(response)
        if final_text is not None:
            logger.info("NodeLLM routing decision: final")
            return {
                "messages": [AIMessage(content=final_text)],
                "final_response": [final_text],
                "next_action": "final",
                "search_required": [False],
                "status_messages": ["complete"],
            }

        request = NodeDriver._extract_tool_request(response)
        if request is None:
            logger.warning("NodeLLM routing decision: backup (invalid tagged response)")
            return {"next_action": "backup", "status_messages": ["generating"]}

        status = "searching through wikipedia" if request["tool"] == "wikisearch" else "searching web"
        if request["tool"] == "node_python":
            status = "computing"
        logger.info("NodeLLM routing decision: %s; request=%s", request["tool"], request)
        return {
            "tool_request": request,
            "tool_execution_history": [{
                "tool": request["tool"],
                "tool_query": request["tool_query"],
                "run_id": state.get("tool_run_id", ""),
            }],
            "next_action": request["tool"],
            "search_required": [True],
            "status_messages": [status],
        }
