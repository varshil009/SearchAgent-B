import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class CloudAgent:
    """LLM client backed by a remote vLLM server (OpenAI-compatible)."""

    def __init__(self):
        server_ip = os.getenv("IPV4")
        if not server_ip:
            raise ValueError("IPV4 environment variable is not set — cannot connect to cloud vLLM server.")

        self.client = OpenAI(
            base_url=f"http://{server_ip}:8000/v1",
            api_key="vllm-placeholder",  # vLLM typically doesn't require a key
        )
        self.model = "meta-llama/Llama-3.1-8B-Instruct"

    def generate_response(self, convo, system_prompt):
        """
        Generate a response from the cloud vLLM server.

        Parameters
        ----------
        convo : list[BaseMessage]
            Conversation messages from LangChain (HumanMessage / AIMessage, etc.)
        system_prompt : str
            System-level instruction prompt.

        Returns
        -------
        str
            The model's response content.
        """
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        # Convert LangChain message objects to OpenAI-format dicts
        for message in convo:
            role = "user" if message.type == "human" else "assistant"
            messages.append({
                "role": role,
                "content": message.content,
            })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )

        return response.choices[0].message.content