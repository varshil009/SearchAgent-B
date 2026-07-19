from services.exa import ExaClient
from .agent_state import AgentState
from langchain_core.messages import ToolMessage

class NodeExaWeb:
    def __init__(self):
        self.client = ExaClient()
    
    def search(self, state:AgentState):
        results = self.client.search(state["search_query"])
        return { 
                "search_results" : results, 
                "search_required" : False,
                "search_query" : ""
                }