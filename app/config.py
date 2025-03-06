import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables from .env file
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client["grant_ai"]  # Database name

# Collections
grants_collection = db["grants"]  # Stores full grant applications
embeddings_collection = db["grant_embeddings"]  # Stores vector embeddings
