from flask import request, jsonify
from app import app
from app.config import embeddings_collection
from app.embeddings import get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from app.config import OPENAI_API_KEY



@app.route("/")
def home():
    return jsonify({"message": "Grant RAG API is running!"})

@app.route("/store_grant", methods=["POST"])
def store_grant():
    """API to store grant-related knowledge with embeddings in MongoDB."""
    data = request.json
    title = data.get("title")
    content = data.get("content")

    if not title or not content:
        return jsonify({"error": "Title and Content are required"}), 400

    embedding = get_embedding(content)  # Generate embedding

    grant_doc = {
        "title": title,
        "content": content,
        "embedding": embedding
    }

    embeddings_collection.insert_one(grant_doc)

    docs = list(embeddings_collection.find({}, {"embedding": 1}))
    print([len(doc["embedding"]) for doc in docs])  # All values should be the same (e.g., 1536 for text-embedding-ada-002)


    return jsonify({"message": "Grant knowledge stored successfully", "title": title})


@app.route("/retrieve_grant", methods=["POST"])
def retrieve_grant():
    """API to retrieve the most relevant grant knowledge using similarity search."""
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = np.array(get_embedding(query))  # Convert to NumPy array

    # Fetch stored embeddings from MongoDB
    stored_grants = list(embeddings_collection.find({}, {"_id": 0, "title": 1, "content": 1, "embedding": 1}))

    if not stored_grants:
        return jsonify({"error": "No grant knowledge available"}), 404

    # Ensure correct format
    stored_embeddings = np.array([np.array(doc["embedding"], dtype=np.float32) for doc in stored_grants])

    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], stored_embeddings)[0]

    # Get the most relevant document
    best_match_index = np.argmax(similarities)
    best_match = stored_grants[best_match_index]

    return jsonify({
        "query": query,
        "retrieved_title": best_match["title"],
        "retrieved_content": best_match["content"],
        "similarity_score": float(similarities[best_match_index])
    })


# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

@app.route("/ask_question", methods=["POST"])
def ask_question():
    data = request.json
    query = data.get("query")

    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Generate query embedding
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)

    # Fetch all stored embeddings
    stored_docs = list(embeddings_collection.find({}, {"_id": 0, "title": 1, "content": 1, "embedding": 1}))
    if not stored_docs:
        return jsonify({"error": "No knowledge stored"}), 404

    # Calculate similarity
    stored_embeddings = np.array([doc["embedding"] for doc in stored_docs])
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]

    # Find best match
    best_match_index = np.argmax(similarities)
    best_match = stored_docs[best_match_index]

    # Prepare prompt for LLM
    prompt = (
        f"Answer the question based on the following information:\n\n"
        f"Information: {best_match['content']}\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    # Generate response from OpenAI
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You provide concise and accurate answers based only on provided information."},
            {"role": "user", "content": prompt}
        ]
    )

    llm_answer = completion.choices[0].message.content.strip()

    return jsonify({
        "query": query,
        "source_title": best_match["title"],
        "source_content": best_match["content"],
        "answer": llm_answer,
        "similarity_score": float(similarities[best_match_index])
    })