import os
import requests
from openai import OpenAI
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)



# usage of client
"""
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct",
    temperature=0,
    messages=[{"role": "system", "content": "Extract entities and relationships from the following text."}, 
              {"role": "user", "content": "John is a software engineer at TechCorp."}]
)
"""