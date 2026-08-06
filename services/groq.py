import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL")

    def generate_response(self, convo, system_prompt):

        messages = [
            {"role": "system", "content": system_prompt}
        ]

        ## messages are created in groq supported formate;
        ## [{"role" : "user", "content" : "asdfghjk..."}, 
        ##  {"role" : "assistant", "content" : "asdfghjk..."]
        for message in convo:
            role = "user" if message.type == "human" else "assistant"

            messages.append({
                "role": role,
                "content": message.content
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
