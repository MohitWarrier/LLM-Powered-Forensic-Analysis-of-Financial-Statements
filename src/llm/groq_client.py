# src/llm/groq_client.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env
load_dotenv()

# Create client with Groq base URL
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

def chat_with_groq(prompt: str, model: str = "llama3-8b-8192") -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()

