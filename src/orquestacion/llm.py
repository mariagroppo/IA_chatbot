import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
print(os.getenv("OPENAI_API_KEY"))

# Initialize client (uses environment variable OPENAI_API_KEY)
client = OpenAI()
""" Sends the prompt to OpenAI and returns the generated response ----------------------------------------------"""
def generate_answer(prompt: str) -> str:

    response = client.responses.create(
        model="gpt-4.1-mini",  # fast + cheap model
        input=prompt
    )

    return response.output[0].content[0].text.strip()
