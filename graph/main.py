from langgraph.graph import START, END, StateGraph, MessagesState
from .node_llm import NodeLLM
from .node_websearch import NodeExaWeb
from .node_wiki import NodeWiki
from .node_response_formatter import NodeFinal
from .node_memory_updater import NodeMemoryUpdater
from .node_title_gen import NodeTitleGen
from .agent_state import AgentState

class Graph:
    def __init__(self):
        self.llm_node = NodeLLM()
        self.websearch_node = NodeExaWeb()
        self.wiki_node = NodeWiki()
        self.final_node = NodeFinal()
        self.memory_node = NodeMemoryUpdater()
        self.title_gen_node = NodeTitleGen()

    def route_after_llm(self, state: AgentState):
        if state["search_required"][-1]:
            # The LLM node stores the tool name in search_queries metadata;
            # we check the status_messages to determine which tool was selected.
            # If the last status_message says "searching through wikipedia", route to wiki.
            if state.get("status_messages") and "wikipedia" in state["status_messages"][-1]:
                return "wikisearch"
            return "websearch"
        return "direct"

    def compileX(self, checkpointer=None):

        graph = StateGraph(AgentState)
        graph.add_node("NodeTitleGen", self.title_gen_node.generate)
        graph.add_node("NodeLLM", self.llm_node.generate)
        graph.add_node("NodeExaWeb", self.websearch_node.search)
        graph.add_node("NodeWiki", self.wiki_node.search)
        graph.add_node("NodeFinal", self.final_node.generate)
        graph.add_node("NodeMemoryUpdater", self.memory_node.generate)
        
        graph.add_edge(START, "NodeTitleGen")
        graph.add_edge("NodeTitleGen", "NodeLLM")
        # on 2nd arg's (self.tool_bool) values are used as keys in next dict arg, 
        # and according to value it directs to that node
        graph.add_conditional_edges(
            "NodeLLM", 
            self.route_after_llm, 
            {
                "direct" : "NodeMemoryUpdater",
                "websearch" : "NodeExaWeb",
                "wikisearch" : "NodeWiki"
            }
        )

        graph.add_edge("NodeExaWeb", "NodeFinal")
        graph.add_edge("NodeWiki", "NodeFinal")
        graph.add_edge("NodeFinal", "NodeMemoryUpdater")
        graph.add_edge("NodeMemoryUpdater", END)

        return graph.compile(checkpointer=checkpointer)