import os
from dotenv import load_dotenv
from pymongo import MongoClient


# Load environment variables
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY2")


# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["grant_ai"]  # Database name

# Collections
grants_collection = db["grants_metadata"]  # Stores full grant applications
embeddings_collection = db["grant_embeddings"]  # Stores vector embeddings
template_collection= db["template"]
