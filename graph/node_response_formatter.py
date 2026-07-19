from services.groq import GroqClient 
from .agent_state import AgentState
from langchain_core.messages import AIMessage

class NodeFinal:
    def __init__(self):
        self.llm = GroqClient()

    def generate(self, state:AgentState):
        
        if state["search_results"]:
            self.tool_results = state["search_results"]
        else:
            self.tool_results = "No tool was needed, generate final response"

        self.node_prompt = f"""
                            Using chat history and tool call response, 
                            create a final answer in a "MARKDOWN" format.
                            This call contains tool call results;
                            And if no tool results are here then consider, to 
                            generate answer tool call is not required.
                            Here are tool results :
                            {self.tool_results}
                            """

        

        response = self.llm.generate_response(state["messages"], self.node_prompt) 
        return {
                "messages" : [AIMessage(content = response)],
                "final_response" : response,
                "search_required": False
                }
    