from . import instrumentation
from langgraph.graph import START, END, StateGraph

from .agent_state import AgentState
from .node_backup import NodeBackup
from .node_llm import NodeLLM
from .node_memory_updater import NodeMemoryUpdater
from .node_python import NodePython
from .node_driver import NodeDriver
from .node_title_gen import NodeTitleGen
from .node_websearch import NodeExaWeb
from .node_wiki import NodeWiki


class Graph:
    def __init__(self):
        self.llm_node = NodeLLM()
        self.websearch_node = NodeExaWeb()
        self.wiki_node = NodeWiki()
        self.python_node = NodePython()
        self.driver_node = NodeDriver()
        self.backup_node = NodeBackup()
        self.memory_node = NodeMemoryUpdater()
        self.title_gen_node = NodeTitleGen()

    @staticmethod
    def route_after_start(state: AgentState):
        """Generate a title only when this conversation does not have one."""
        return "title_gen" if not state.get("convo_title", "").strip() else "llm"

    @staticmethod
    def route_after_decision(state: AgentState):
        """NodeDriver is the sole owner of the graph's next-action decision."""
        return state.get("next_action", "backup")

    def compileX(self, checkpointer=None):
        graph = StateGraph(AgentState)
        graph.add_node("NodeTitleGen", self.title_gen_node.generate)
        graph.add_node("NodeLLM", self.llm_node.generate)
        graph.add_node("NodeExaWeb", self.websearch_node.search)
        graph.add_node("NodeWiki", self.wiki_node.search)
        graph.add_node("NodePython", self.python_node.search)
        graph.add_node("NodeDriver", self.driver_node.generate)
        graph.add_node("NodeBackup", self.backup_node.generate)
        graph.add_node("NodeMemoryUpdater", self.memory_node.generate)

        graph.add_conditional_edges(
            START,
            self.route_after_start,
            {"title_gen": "NodeTitleGen", "llm": "NodeLLM"},
        )
        graph.add_edge("NodeTitleGen", "NodeLLM")
        graph.add_conditional_edges(
            "NodeLLM",
            self.route_after_decision,
            {
                "final": "NodeMemoryUpdater",
                "websearch": "NodeExaWeb",
                "wikisearch": "NodeWiki",
                "node_python": "NodePython",
                "backup": "NodeBackup",
            },
        )

        graph.add_conditional_edges(
            "NodeDriver",
            self.route_after_decision,
            {
                "final": "NodeMemoryUpdater",
                "websearch": "NodeExaWeb",
                "wikisearch": "NodeWiki",
                "node_python": "NodePython",
                "decide": "NodeDriver",
                "backup": "NodeBackup",
            },
        )
        graph.add_edge("NodeExaWeb", "NodeDriver")
        graph.add_edge("NodeWiki", "NodeDriver")
        graph.add_edge("NodePython", "NodeDriver")
        graph.add_edge("NodeBackup", "NodeMemoryUpdater")
        graph.add_edge("NodeMemoryUpdater", END)

        return graph.compile(checkpointer=checkpointer)
