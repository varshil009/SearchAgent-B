from langgraph.graph import MessagesState

class AgentState(MessagesState):
    search_required : bool
    search_query : str
    search_results : list
    final_response : str