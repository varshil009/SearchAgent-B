from graph.main import Graph
from graph.agent_state import AgentState

class AgentLoop:
    def __init__(self):
        self.graph = Graph().compileX()

    def run(self, state:AgentState):
        final_results = self.graph.invoke(state)
        return final_results

    def stream(self, state: AgentState):
        yield from self.graph.stream(state, stream_mode="updates")
