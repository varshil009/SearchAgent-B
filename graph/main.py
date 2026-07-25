from langgraph.graph import START, END, StateGraph, MessagesState
from .node_llm import NodeLLM
from .node_websearch import NodeExaWeb
from .node_response_formatter import NodeFinal
from .node_memory_updater import NodeMemoryUpdater
from .agent_state import AgentState

class Graph:
    def __init__(self):
        self.llm_node = NodeLLM()
        self.websearch_node = NodeExaWeb()
        self.final_node = NodeFinal()
        self.memory_node = NodeMemoryUpdater()

    def route_after_llm(self, state: AgentState):
        if state["search_required"][-1]:
            return "websearch"
        return "direct"

    def compileX(self):

        graph = StateGraph(AgentState)
        graph.add_node("NodeLLM", self.llm_node.generate)
        graph.add_node("NodeExaWeb", self.websearch_node.search)
        graph.add_node("NodeFinal", self.final_node.generate)
        graph.add_node("NodeMemoryUpdater", self.memory_node.generate)
        graph.add_edge(START, "NodeLLM")
        # on 2nd arg's (self.tool_bool) values are used as keys in next dict arg, 
        # and according to value it directs to that node
        graph.add_conditional_edges(
            "NodeLLM", 
            self.route_after_llm, 
            {
                "direct" : "NodeMemoryUpdater",
                "websearch" : "NodeExaWeb"
            }
        )

        graph.add_edge("NodeExaWeb", "NodeFinal")
        graph.add_edge("NodeFinal", "NodeMemoryUpdater")
        graph.add_edge("NodeMemoryUpdater", END)

        return graph.compile()
