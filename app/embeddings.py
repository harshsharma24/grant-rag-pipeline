import hashlib
import numpy as np
from langchain_openai import ChatOpenAI
from app.config import OPENAI_API_KEY

# Initialize GPT-4o-Mini model from OpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

def get_embedding(text):
    """Generate pseudo-embeddings using GPT-4o-Mini."""
    response = llm.invoke(f"Extract the key concepts from this text: {text}")
    key_phrases = response.content.split()

    hashed_values = []
    for key_phrase in key_phrases:
        hashed_value = int(hashlib.md5(key_phrase.encode()).hexdigest(), 16) % 1000
        hashed_values.append(hashed_value)

    # Convert to NumPy array for normalization
    hashed_values = np.array(hashed_values)

    # Normalize vector to unit length
    embedding = hashed_values / np.linalg.norm(hashed_values)

    return embedding.tolist()
