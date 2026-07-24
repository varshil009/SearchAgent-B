from agent_loop import AgentLoop
from langchain_core.messages import HumanMessage

print(AgentLoop().run(
    {
        "messages" : [HumanMessage(content="give me follow up on recent ODI series between INDIA and ENGLAND")],
        "search_required" : False,
        "search_results" : None,
        "search_query" : "",
        "final_response" : ""
    }))

