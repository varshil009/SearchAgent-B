from langgraph.graph import MessagesState
from operator import add
from typing import Annotated

class AgentState(MessagesState):
    """
    Agent state, passed through each node;
    messages is a default must key;
    heres other payload that is required for process
    """
    search_required: Annotated[list[bool], add]
    search_queries: Annotated[list[str], add]
    image_links: Annotated[list[str], add]
    status_messages: Annotated[list[str], add]
    convo_memory: str
    search_results : list
    final_response: Annotated[list[str], add]
    conversation_title: str