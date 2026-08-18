from services.groq import GroqClient
from services.app_logger import get_app_logger
from langchain_core.messages import HumanMessage
from .think_tag import _strip_think_tags as filterThinkTags
from .agent_state import AgentState

logger = get_app_logger()

TITLE_GEN_PROMPT = """
You are a conversation title generator. Given a user's first query in a conversation,
generate a short, descriptive title (max 5-6 words) that captures the topic.

Rules:
- Output ONLY the title, nothing else. No quotes, no punctuation, no explanation.
- The title should be concise and meaningful.
- Examples: "Web Search Agent Setup", "Latest AI Research Papers", "Weather in Tokyo"

User query: {query}
"""


class NodeTitleGen:
    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        messages = state.get("messages", [])
        if not messages:
            return {"conversation_title": "New Chat"}

        first_query = messages[0].content if messages else ""
        query = first_query.strip()

        # If query is short (< 50 chars), return it as-is (no API call)
        if len(query) < 50:
            logger.info("Query too short for title gen (%d chars), using query as title", len(query))
            return {"conversation_title": query}

        # Otherwise, call Groq for title generation
        try:
            prompt = TITLE_GEN_PROMPT.format(query=query)
            response = self.llm.generate_response(
                [HumanMessage(content=query)],
                prompt
            )
            title = response.strip().strip('"').strip("'")
            if title:
                title = filterThinkTags(title)
                logger.info("Generated title: %s", title)
                return {"conversation_title": title}
        except Exception as e:
            logger.error("Title generation failed: %s", e)

        # Fallback: first 100 chars of query
        fallback = query[:100]
        logger.info("Fallback title: %s", fallback)
        return {"conversation_title": fallback}
