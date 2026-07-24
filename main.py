from agent_loop import AgentLoop
from langchain_core.messages import HumanMessage

print(AgentLoop().run(
    {
        "messages" : [HumanMessage(content="What is this paper leak studenst protests going on in India")],
        "search_required" : [False],
        "search_results" : None,
        "search_queries" : [],
        "final_response" : ""
    }))

