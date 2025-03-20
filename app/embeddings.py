import openai
import os
from dotenv import load_dotenv
from app.config import OPENAI_API_KEY


client = openai.OpenAI(api_key=OPENAI_API_KEY)

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-ada-002",  # explicit embedding model
        input=text
    )
    return response.data[0].embedding
