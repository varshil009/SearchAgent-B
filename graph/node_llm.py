from json.decoder import JSONDecodeError
from services.groq import GroqClient 
import json
from .agent_state import AgentState
from langchain_core.messages import AIMessage
import ast
import time

class NodeLLM:
    def __init__(self):
        with open(r"graph/tools.json", "r", encoding="utf-8") as f:
            tools = json.load(f)

        with open(r"graph/tool_request.json", "r", encoding="utf-8") as f:
            tools_request_format = json.load(f)

        self.date = time.strftime("%Y-%m-%d %H:%M:%S")
        self.node_prompt = f"""
                                You are a chatbot.

                                Analyze the user's query and respond best according to your knowledge.

                                If the user query requires latest developement/news,
                                go through list of tools and output strictly json.
                                and all json keys must be strictly double quoted or "key".
                                
                                Query initiation date : {self.date}

                                list of tools : 
                                {tools}

                                STRICLY response format if tool call is required :
                                {tools_request_format}

                                You will find conversation history along, generate answer accordingly.
                            """
        
        self.llm = GroqClient()

    def generate(self, state: AgentState):
        prompt_with_memory = f"""
            {self.node_prompt}

            Conversation memory from earlier API calls:
            {state.get("convo_memory") or "No saved memory yet."}
        """
        response = self.llm.generate_response(
            state["messages"],
            prompt_with_memory
        )

        # First try proper JSON
        try:
            parsed_response = json.loads(response)

        # If not JSON, try Python dict syntax
        except JSONDecodeError:
            try:
                parsed_response = ast.literal_eval(response)

            # It's neither JSON nor a Python dict → normal LLM answer
            except (ValueError, SyntaxError):
                return self.extract_json_and_return(response)

        # Tool call
        if parsed_response.get("tool_required"):
            return {
                "search_required": [True],
                "search_queries": [parsed_response["tool_query"]],
                "status_messages": ["searching web"]
            }

        # Valid JSON/dict, but no tool required
        return {
            "search_required": [False]
        }

    
    def extract_json_and_return(self, response):
        try:
            # Extract JSON object from surrounding conversational text
            start = response.find("{")
            end = response.rfind("}") + 1

            if start == -1 or end == 0:
                raise ValueError("No JSON found")

            json_string = response[start:end]

            # Try valid JSON first
            try:
                r = json.loads(json_string)

            # Fallback for Python-style dict from LLM
            except json.JSONDecodeError:
                r = ast.literal_eval(json_string)

            if r.get("tool_required"):
                return {
                    "search_required": [True],
                    "search_queries": [r["tool_query"]],
                    "status_messages": ["searching web"]
                }

        except (ValueError, SyntaxError, json.JSONDecodeError):
            pass

        # No valid tool call → treat as normal response
        return {
            "messages": [AIMessage(content=response)],
            "search_required": [False],
            "final_response": [response]
        }
