from langgraph.graph import MessagesState
from operator import add
from typing import Annotated

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
    search_results : list
    final_response: Annotated[list[str], add]
    conversation_title: str