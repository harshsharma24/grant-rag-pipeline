from flask import request, jsonify
from app import app
from app.config import grants_collection, embeddings_collection
from app.embeddings import get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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

@app.route("/store_knowledge", methods=["POST"])
def store_knowledge():
    """API to store knowledge-based text data for RAG."""
    data = request.json
    title = data.get("title")
    content = data.get("content")

    if not title or not content:
        return jsonify({"error": "Title and Content are required"}), 400

    embedding = get_embedding(content)  # Generate embedding

    knowledge_doc = {
        "title": title,
        "content": content,
        "embedding": embedding
    }

    embeddings_collection.insert_one(knowledge_doc)

    return jsonify({"message": "Knowledge stored successfully", "title": title})

@app.route("/retrieve_knowledge", methods=["POST"])
def retrieve_knowledge():
    """API to retrieve the most relevant stored knowledge using similarity search."""
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = np.array(get_embedding(query))  # Ensure it's a NumPy array

    # Fetch all stored embeddings from MongoDB
    stored_knowledge = list(embeddings_collection.find({}, {"_id": 0, "title": 1, "content": 1, "embedding": 1}))

    if not stored_knowledge:
        return jsonify({"error": "No knowledge data available"}), 404

    # 🔹 Debug: Print shapes of stored embeddings
    for doc in stored_knowledge:
        print(f"Title: {doc['title']}, Embedding Shape: {np.array(doc['embedding']).shape}")

    try:
        # Convert stored embeddings into NumPy array (Ensures same shape)
        stored_embeddings = np.array([np.array(doc["embedding"], dtype=np.float32) for doc in stored_knowledge])

    except ValueError as e:
        print("❌ ValueError Detected:", e)
        return jsonify({"error": "Inconsistent embedding shapes in database."}), 500

    # Compute cosine similarity between query embedding and stored embeddings
    similarities = cosine_similarity([query_embedding], stored_embeddings)[0]

    # Get the most relevant document
    best_match_index = np.argmax(similarities)
    best_match = stored_knowledge[best_match_index]

    return jsonify({
        "query": query,
        "retrieved_title": best_match["title"],
        "retrieved_content": best_match["content"],
        "similarity_score": float(similarities[best_match_index])
    })
