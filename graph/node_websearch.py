from services.exa import ExaClient
from services.app_logger import get_app_logger
from .agent_state import AgentState
from .tool_result_schema import describe_tool_result


logger = get_app_logger("websearch")
MAX_SUMMARY_WORDS = 120
MAX_HIGHLIGHT_WORDS = 40
MAX_HIGHLIGHTS = 3


def _truncate_words(value, limit: int):
    """Keep Exa's free-form text within the decision node's prompt budget."""
    if not isinstance(value, str):
        return value
    words = value.split()
    if len(words) <= limit:
        return value
    return " ".join(words[:limit]) + " …"


def _compact_result(result: dict) -> dict:
    highlights = result.get("highlights")
    if isinstance(highlights, list):
        highlights = [
            _truncate_words(str(highlight), MAX_HIGHLIGHT_WORDS)
            for highlight in highlights[:MAX_HIGHLIGHTS]
        ]
    else:
        highlights = _truncate_words(highlights, MAX_HIGHLIGHT_WORDS)

    return {
        "title": _truncate_words(result.get("title"), 30),
        "published_date": result.get("published_date"),
        "author": _truncate_words(result.get("author"), 30),
        "summary": _truncate_words(result.get("summary"), MAX_SUMMARY_WORDS),
        "highlights": highlights,
    }

class NodeExaWeb:
    def __init__(self):
        self.client = ExaClient()
    
    def search(self, state:AgentState):
        ## for now to limit token exceeding, cap to 3 results only
        try:
            query = state["tool_request"]["tool_query"]
            raw_results = self.client.search(query)[:3]
            results = [_compact_result(result) for result in raw_results]
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
