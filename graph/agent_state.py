from langgraph.graph import MessagesState
from operator import add
from typing import Annotated, Any

class AgentState(MessagesState):
    """
    Agent state, passed through each node;
    messages is a default must key;
    heres other payload that is required for process

    there are 2 uses of Annotated here, 
    first one is using it as data validation, add a wrong type value and it raises error
    operator.add is used as reducer; whenever a node returns a value without this operator.add
        it have to do like ```return {"key" : state["key"] + [current_value]}```; 
        with reducer you can jsut ```return {"key" : [current_Value]}```
    """
    search_required: Annotated[list[bool], add]
    search_queries: Annotated[list[str], add]
    image_links: Annotated[list[str], add]
    status_messages: Annotated[list[str], add]
    convo_memory: str
    # This is deliberately a non-reduced field: Python only sees the newest
    # tool output, never the entire historical state.
    tool_results: Any
    latest_tool_schema: str
    tool_request: dict[str, Any]
    tool_execution_history: Annotated[list[dict[str, str]], add]
    tool_run_id: str
    loop_blocked: bool
    next_action: str
    final_response: Annotated[list[str], add]
    convo_title: str
    conversation_title: str
