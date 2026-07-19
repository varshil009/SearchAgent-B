import os
from groq import Groq

class GroqClient:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL")

    def generate_response(self, convo, system_prompt):

        messages = [
            {"role": "system", "content": system_prompt}
        ]

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