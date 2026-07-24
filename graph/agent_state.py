from langgraph.graph import MessagesState
from operator import add
from typing import Annotated

class AgentState(MessagesState):
    search_required: Annotated[list[bool], add]
    search_queries: Annotated[list[str], add]
    search_results : list
    final_response : str
