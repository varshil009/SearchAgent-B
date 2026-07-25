from graph.main import Graph
from graph.agent_state import AgentState
from services.app_logger import get_app_logger
from langgraph.checkpoint.memory import MemorySaver


logger = get_app_logger()

class AgentLoop:
    def __init__(self):
        self.graph = Graph().compileX()
        self.thread_memories: dict[str, MemorySaver] = {}

    def _get_memory(self, thread_id: str) -> MemorySaver:
        """Get or create a MemorySaver for the given thread."""
        if thread_id not in self.thread_memories:
            self.thread_memories[thread_id] = MemorySaver()
            logger.info("Created new MemorySaver for thread=%s", thread_id)
        return self.thread_memories[thread_id]

    def run(self, state: AgentState, thread_id: str | None = None):
        memory = self._get_memory(thread_id) if thread_id else MemorySaver()
        config = {"configurable": {"thread_id": thread_id or "default"}}
        final_results = self.graph.invoke(state, config=config)
        return final_results

    def stream(self, state: AgentState, thread_id: str | None = None):
        memory = self._get_memory(thread_id) if thread_id else MemorySaver()
        config = {"configurable": {"thread_id": thread_id or "default"}}
        for update in self.graph.stream(state, config=config, stream_mode="updates"):
            node_update = next(iter(update.values()))
            yield update