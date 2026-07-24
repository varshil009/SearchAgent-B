from services.exa import ExaClient
from .agent_state import AgentState
from langchain_core.messages import ToolMessage

class NodeExaWeb:
    def __init__(self):
        self.client = ExaClient()
    
    def search(self, state:AgentState):
        ## for now to limit token exceeding, cap to 3 results only
        results = self.client.search(state["search_queries"][-1])[:3]

        return { 
                "search_results" : results, 
                "search_required" : [False]
                }
