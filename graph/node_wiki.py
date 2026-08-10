from services.wikipedia import WikiClient
from .agent_state import AgentState
from .tool_result_schema import describe_tool_result

class NodeWiki:
    def __init__(self):
        self.client = WikiClient()

    def search(self, state: AgentState):
        try:
            query = state["tool_request"]["tool_query"]
            raw_results = self.client.search(query)[:5]
            results = []
            for article in raw_results:
                words = article.split()
                truncated = " ".join(words[:100])  # cap each article at 100 words
                results.append(truncated)
        except Exception as error:
            logger = None
            try:
                from services.app_logger import get_app_logger
                logger = get_app_logger("wikisearch")
                logger.warning("Wikipedia search failed: %s", error)
            except Exception:
                pass
            results = [
                {
                    "error": (
                        "Wikipedia search is temporarily unavailable. Do not claim that "
                        "historical information was verified."
                    )
                }
            ]

        return {
            "tool_results": results,
            "latest_tool_schema": describe_tool_result(results),
            "search_required": [False],
            "status_messages": ["generating"]
        }
