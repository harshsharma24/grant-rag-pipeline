from flask import request, jsonify
from app import app
from app.config import grants_collection, embeddings_collection
from app.embeddings import get_embedding

@app.route("/")
def home():
    return jsonify({"message": "Grant RAG API is running!"})

@app.route("/store_embedding", methods=["POST"])
def store_embedding():
    """API to generate and store embeddings in MongoDB."""
    data = request.json
    text = data.get("text")

    if not text:
        return jsonify({"error": "Text is required"}), 400

    embedding = get_embedding(text)

    # Store in MongoDB
    embeddings_collection.insert_one({"text": text, "embedding": embedding})

    return jsonify({"message": "Embedding stored successfully", "text": text})

@app.route("/get_embeddings", methods=["GET"])
def get_embeddings():
    """API to retrieve all stored embeddings."""
    embeddings = list(embeddings_collection.find({}, {"_id": 0}))
    return jsonify({"embeddings": embeddings})
