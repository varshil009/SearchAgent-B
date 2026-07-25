from langgraph.graph import MessagesState
from operator import add
from typing import Annotated

class AgentState(MessagesState):
    search_required: Annotated[list[bool], add]
    search_queries: Annotated[list[str], add]
    image_links: Annotated[list[str], add]
    status_messages: Annotated[list[str], add]
    convo_memory: str
    search_results : list
    final_response: Annotated[list[str], add]
