from services.exa import ExaClient
from services.app_logger import get_app_logger
from .agent_state import AgentState
from .tool_result_schema import describe_tool_result


logger = get_app_logger("websearch")
LLM_RESULT_FIELDS = ("title", "published_date", "author", "summary", "highlights")

class NodeExaWeb:
    def __init__(self):
        self.client = ExaClient()
    
    def search(self, state:AgentState):
        ## for now to limit token exceeding, cap to 3 results only
        try:
            raw_results = self.client.search(state["search_queries"][-1])[:3]
            results = [
                {field: result.get(field) for field in LLM_RESULT_FIELDS}
                for result in raw_results
            ]
            image_links = [
                result["image"]
                for result in raw_results
                if result.get("image")
            ]
        except Exception as error:
            logger.warning("Web search failed: %s", error)
            results = [
                {
                    "error": (
                        "Web search is temporarily unavailable. Do not claim that "
                        "current information was verified."
                    )
                }
            ]
            image_links = []

        return {
                "tool_results": results,
                "latest_tool_schema": describe_tool_result(results),
                "image_links": image_links,
                "search_required" : [False],
                "status_messages": ["generating"]
                }
