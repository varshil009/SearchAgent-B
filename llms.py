from services.groqx import GroqClient
from dotenv import load_dotenv
from groq import Groq
from pathlib import Path
import os

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

class GroqModels:
    def __init__(self, curr_model = None):
        if not curr_model:
            self.model = "qwen/qwen3.6-27b"
        else:
            self.model = curr_model

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, query, model):
        messages = [
            {"role": "system", "content": "answer the question"}
        ]

        ## messages are created in groq supported formate;
        ## [{"role" : "user", "content" : "asdfghjk..."}, 
        ##  {"role" : "assistant", "content" : "asdfghjk..."]
        
        messages.append({
            "role": "user",
            "content": query
        })

        response = self.client.chat.completions.create(
            model=model,
            messages=messages
        )

        return response.choices[0].message.content


models = [
        "qwen/qwen3.6-27b"
        "groq/compound",
        "groq/compound-mini",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-safeguard-20b",
                    ]

client = GroqModels()
response = []
for x in models:
    time.sleep(1)
    try:
        print("===============================================================")
        print("Model Name :", x)
        print("===============================================================")
        ans = client.generate("Hello how are ya", x)
        response.append(ans)
        print(ans)
    except Exception as E:
        print("===============================================================")
        print("Model Name :", x)
        print("===============================================================")
        #ans = client.generate("Hello how are ya", x)
        response.append(E)
        print(E)

import time
txt = ""
for model, ans in zip(models, ans):
    txt += "\n===============================================================\n"
    txt += "Model Name :" + model
    txt += "\n===============================================================\n"
    txt += ans
    txt += "\nxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n"
with open("model_responses.txt", "w") as f:
    f.write(txt)