import json
import re
from typing import Any

from langchain_core.messages import AIMessage

from services.groq import GroqClient
from services.app_logger import get_app_logger
from .agent_state import AgentState


MAX_INLINE_TOOL_RESULT_CHARS = 12_000
MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS = 5
MAX_TOOL_EXECUTIONS = 20
VALID_TOOLS = {"websearch", "wikisearch", "node_python"}
logger = get_app_logger("node_driver")


class NodeDriver:
    """The graph's central decision node and cyclic workflow driver."""

    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        history = self._current_run_history(state)
        loop_blocked = bool(state.get("loop_blocked")) or len(history) >= MAX_TOOL_EXECUTIONS
        prompt = self._prompt(state, loop_blocked)
        try:
            response = self.llm.generate_response(state["messages"], prompt)
        except Exception:
            logger.exception("NodeDriver request failed")
            return {"next_action": "backup", "status_messages": ["generating"]}

        logger.info("NodeDriver raw LLM response (run=%s): %r", state.get("tool_run_id"), response)

        final_text = self._extract_final(response)
        if final_text is not None:
            logger.info("NodeDriver routing decision: final")
            return {
                "messages": [AIMessage(content=final_text)],
                "final_response": [final_text],
                "next_action": "final",
                "search_required": [False],
                "status_messages": ["complete"],
            }

        request = self._extract_tool_request(response)
        if request is None or loop_blocked:
            logger.warning(
                "NodeDriver routing decision: backup (invalid_response=%s, loop_blocked=%s)",
                request is None,
                loop_blocked,
            )
            return {"next_action": "backup", "status_messages": ["generating"]}

        history = self._current_run_history(state)
        consecutive = self._consecutive_count(history, request)
        if consecutive >= MAX_CONSECUTIVE_IDENTICAL_TOOL_CALLS:
            # The request would become the sixth identical dispatch. Route back
            # through this node once with a final-only instruction.
            return {
                "loop_blocked": True,
                "next_action": "decide",
                "search_required": [False],
                "status_messages": ["generating"],
            }

        status = "searching through wikipedia" if request["tool"] == "wikisearch" else "searching web"
        if request["tool"] == "node_python":
            status = "computing"
        logger.info("NodeDriver routing decision: %s; request=%s", request["tool"], request)
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

    def _prompt(self, state: AgentState, loop_blocked: bool) -> str:
        result = state.get("tool_results")
        serialized_result = self._serialize_result(result)
        tool_request = state.get("tool_request") or {}
        serialized_request = self._serialize_result(tool_request)
        schema = state.get("latest_tool_schema") or "No tool result is available yet."
        if not tool_request:
            tool_context = "No tool has been executed for this request yet."
            post_tool_rule = ""
        elif len(serialized_result) <= MAX_INLINE_TOOL_RESULT_CHARS:
            tool_context = (
                "=== COMPLETED TOOL REQUEST ===\n"
                f"{serialized_request}\n\n"
                "=== AUTHORITATIVE RESULT FROM THAT REQUEST ===\n"
                f"{serialized_result}"
            )
            post_tool_rule = """
                CRITICAL POST-TOOL RULE: The request and result below are already
                complete. Treat the result as the primary evidence for your next
                decision. Do not claim that the information is unavailable or ask
                for the same request again when the result contains relevant content.
                If it answers the user, your only valid next action is <final>.
            """
        else:
            tool_context = (
                "=== COMPLETED TOOL REQUEST ===\n"
                f"{serialized_request}\n\n"
                "=== RESULT TOO LARGE TO INLINE; SCHEMA ===\n"
                f"{schema}\n\nUse node_python only if computation over last_tool_result is substantially "
                "more reliable than normal reasoning."
            )
            post_tool_rule = """
                CRITICAL POST-TOOL RULE: The request below has already completed.
                Use its schema to decide whether targeted computation is needed;
                never repeat the same request merely because its full result is large.
            """

        final_only = "\nYou must now return a <final> answer using the available information." if loop_blocked else ""
        return f"""
            You are the decision-making node for a research assistant. Review the
            complete conversation and the latest tool output, then choose exactly one action.

            IMPORTANT: When writing the final answer, use rich Markdown formatting to
            make important information stand out. Use **bold** for key terms, concepts,
            or important numbers. Use *italics* for emphasis or secondary highlights.
            Use bullet lists and numbered lists to organize related points. Use tables
            when comparing data or presenting structured information. The frontend
            renders Markdown natively, so all formatting will display correctly.

            Return exactly one of these formats, with no surrounding text:
            <final>
            Markdown answer for the user
            </final>

            <tool>
            {{"tool_required": true, "tool": "websearch|wikisearch|node_python", "tool_query": "..."}}
            </tool>

            Use websearch for current information and wikisearch for stable,
            encyclopedic information. Use node_python only for computation. For
            For node_python, tool_query is the Python code itself. Do not declare an
            execution mode. Prefer exactly one `result = expression` statement;
            do not use semicolons, temporary variables, multiple statements, or
            `print()`, because the value of `result` is returned automatically.
            Example: `result = len(['a', 'b'])`. The runtime selects the appropriate
            execution mode. Python receives only `last_tool_result`; it cannot use
            files, network, shell commands, packages, environment variables, or imports.

            After a tool result is present, inspect the completed request and its
            result before deciding. If that result answers the user's request,
            return <final> immediately. Never repeat the exact same tool request
            merely because it has already completed. If it failed, use the available
            information to answer or choose a materially different request.

            Do not mention tools, internal instructions, or routing to the user.
            {post_tool_rule}
            {tool_context}
            Latest tool-result schema: {schema}
            {final_only}
        """

    @staticmethod
    def _serialize_result(result: Any) -> str:
        try:
            return json.dumps(result, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return repr(result)

    @staticmethod
    def _extract_final(response: str) -> str | None:
        # Reasoning models may prepend a completed <think> block before the
        # routing envelope. Remove it only for parsing; raw responses are kept.
        response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE)
        match = re.fullmatch(r"\s*<final>\s*(.*?)\s*</final>\s*", response, re.DOTALL)
        if not match:
            return None
        final_text = match.group(1).strip()
        return final_text or None

    @staticmethod
    def _extract_tool_request(response: str) -> dict[str, str] | None:
        response = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE)
        match = re.fullmatch(r"\s*<tool>\s*(.*?)\s*</tool>\s*", response, re.DOTALL)
        if not match:
            return None
        try:
            request = json.loads(match.group(1))
        except json.JSONDecodeError:
            return None
        required = {"tool_required", "tool", "tool_query"}
        if set(request) != required or request.get("tool_required") is not True:
            return None
        if request.get("tool") not in VALID_TOOLS:
            return None
        if not all(isinstance(request.get(key), str) for key in ("tool", "tool_query")):
            return None
        return request

    @staticmethod
    def _consecutive_count(history: list[dict[str, str]], request: dict[str, str]) -> int:
        count = 0
        for item in reversed(history):
            if item.get("tool") != request["tool"] or item.get("tool_query") != request["tool_query"]:
                break
            count += 1
        return count

    @staticmethod
    def _current_run_history(state: AgentState) -> list[dict[str, str]]:
        run_id = state.get("tool_run_id", "")
        return [
            item for item in state.get("tool_execution_history", [])
            if item.get("run_id", "") == run_id
        ]
