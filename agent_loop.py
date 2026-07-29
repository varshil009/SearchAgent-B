from graph.main import Graph
from graph.agent_state import AgentState
from services.app_logger import get_app_logger
from langgraph.checkpoint.memory import MemorySaver
from typing import Optional


logger = get_app_logger()

class AgentLoop:
    def __init__(self):
        self.thread_memories: dict[str, MemorySaver] = {}
        self.default_memory = MemorySaver()
        self.graph = Graph().compileX(checkpointer=self.default_memory)

    def has_memory(self, thread_id: Optional[str]) -> bool:
        return bool(thread_id and thread_id in self.thread_memories)

    def _get_graph_for_thread(self, thread_id: Optional[str] = None):
        if not thread_id:
            return self.graph
        if thread_id not in self.thread_memories:
            self.thread_memories[thread_id] = MemorySaver()
            logger.info("Created MemorySaver for thread=%s", thread_id)
        return Graph().compileX(checkpointer=self.thread_memories[thread_id])

    def run(self, state: AgentState, thread_id: Optional[str] = None):
        graph = self._get_graph_for_thread(thread_id)
        config = {"configurable": {"thread_id": thread_id or "default"}}
        final_results = graph.invoke(state, config=config)
        return final_results

    def stream(self, state: AgentState, thread_id: Optional[str] = None):
        graph = self._get_graph_for_thread(thread_id)
        config = {"configurable": {"thread_id": thread_id or "default"}}
        for update in graph.stream(state, config=config, stream_mode="updates"):
            node_update = next(iter(update.values()))
            yield update
