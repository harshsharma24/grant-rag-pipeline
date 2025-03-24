from flask import request, jsonify
from app import app
from app.config import embeddings_collection, template_collection
from app.embeddings import get_embedding
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import openai
from app.config import OPENAI_API_KEY
import docx
import fitz
import os
import tiktoken
from app.grant_service import fetch_grant_details, store_grant_metadata, update_grant_status
import requests
from flask import send_file
from docx import Document
import tempfile


# Initialize OpenAI Client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

@app.route("/")
def home():
    return jsonify({"message": "Grant RAG API is running!"})



def extract_text(file_path):
    """Extract text from DOCX or PDF."""
    if file_path.endswith(".docx"):
        return "\n".join(para.text for para in docx.Document(file_path).paragraphs)
    
    if file_path.endswith(".pdf"):
        return "\n".join(page.get_text() for page in fitz.open(file_path))
    
    return ""

def get_embedding(text):
    """Generate OpenAI embedding."""
    return client.embeddings.create(model="text-embedding-ada-002", input=text).data[0].embedding

tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

def chunk_text(text, chunk_size=250, overlap=50):
    """Splits text into overlapping chunks of specified token size."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)  # Convert tokens back to text
        chunks.append(chunk_text)
        start += chunk_size - overlap  # Move with overlap
    return chunks

def chunk_text_grant(text, chunk_size=5000):
    """Splits text into overlapping chunks of specified token size."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        chunk_tokens = tokens[start : start + chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        start += chunk_size  # No overlap needed
    return chunks

def store_generic_grant(title, file_path):
    """Extract text from the generic grant document and store as a single vector embedding."""
    text = extract_text(file_path)
    
    if not text.strip():
        print(f"❌ No text extracted from {title}")
        return
    
    embedding = get_embedding(text)
    
    if embedding is None:
        print(f"❌ Failed to generate embedding for {title}")
        return

    # Store full document as one vector
    template_collection.insert_one({
        "title": title,
        "content": text,
        "embedding": embedding
    })
    
    print(f"✅ Stored Generic Grant: {title}")



@app.route("/upload_grant", methods=["POST"])
def upload_grant():
    """Upload DOCX/PDF, extract text, chunk, and store in MongoDB."""
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    # Extract text
    text = extract_text(file_path)

    # Remove temp file
    os.remove(file_path)

    if not text.strip():
        return jsonify({"error": "No text extracted"}), 400

    # Chunk the document
    chunks = chunk_text(text)

    # Store each chunk separately with embedding
    stored_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding is None:
            return jsonify({"error": "Failed to generate embeddings"}), 500

        chunk_doc = {
            "title": file.filename,
            "chunk_index": i,
            "content": chunk,
            "embedding": embedding
        }
        embeddings_collection.insert_one(chunk_doc)
        stored_chunks.append(chunk_doc)

    return jsonify({
        "message": "Uploaded successfully",
        "title": file.filename,
        "total_chunks": len(stored_chunks)
    })

@app.route("/upload_generic_grant", methods=["POST"])
def upload_generic_grant():
    """Upload DOCX/PDF generic grant and store as a single or chunked embedding."""
    file = request.files.get("file")

    if not file:
        return jsonify({"error": "No file provided"}), 400
    
    file_path = f"/tmp/{file.filename}"
    file.save(file_path)

    # Extract text
    text = extract_text(file_path)

    # Remove temp file
    os.remove(file_path)

    if not text.strip():
        return jsonify({"error": "No text extracted"}), 400

    # Chunk the document if it exceeds 8192 tokens
    chunks = chunk_text_grant(text) if len(tokenizer.encode(text)) > 8192 else [text]

    stored_chunks = []
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        if embedding is None:
            return jsonify({"error": "Failed to generate embeddings"}), 500

        chunk_doc = {
            "title": file.filename,
            "chunk_index": i,
            "content": chunk,
            "embedding": embedding
        }
        template_collection.insert_one(chunk_doc)
        stored_chunks.append(chunk_doc)

    return jsonify({
        "message": "Generic Grant Uploaded Successfully",
        "title": file.filename,
        "total_chunks": len(stored_chunks)
    })

@app.route("/retrieve_relevant_grants", methods=["POST"])
def retrieve_relevant_grants():
    """Retrieve the most relevant grant chunks using similarity search."""
    data = request.json
    query = data.get("query")
    top_n = data.get("top_n", 5)  # Default to returning top 5 relevant chunks

    if not query:
        return jsonify({"error": "Query is required"}), 400

    query_embedding = np.array(get_embedding(query))

    # ✅ Fetch stored chunks from `embeddings_collection`
    stored_chunks = list(embeddings_collection.find({}, {"_id": 0, "title": 1, "chunk_index": 1, "content": 1, "embedding": 1}))

    if not stored_chunks:
        return jsonify({"error": "No grant knowledge available"}), 404

    # Convert stored embeddings into NumPy arrays
    stored_embeddings = np.array([np.array(doc["embedding"], dtype=np.float32) for doc in stored_chunks])
    
    # Compute cosine similarity
    similarities = cosine_similarity([query_embedding], stored_embeddings)[0]

    # Get indices of top N most similar chunks
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    # Retrieve top N chunks
    top_results = [
        {
            "title": stored_chunks[i]["title"],
            "chunk_index": stored_chunks[i]["chunk_index"],
            "content": stored_chunks[i]["content"],
            "similarity_score": float(similarities[i])
        }
        for i in top_indices
    ]

    return jsonify({
        "query": query,
        "top_matches": top_results
    })




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

@app.route("/fetch_and_store_grant", methods=["POST"])
def fetch_and_store_grant():
    """Fetch full grant details from Grants.gov API and return the full response."""
    data = request.json
    opportunity_id = data.get("opportunityId")

    GRANTS_API_URL = "https://api.grants.gov/v1/api/fetchOpportunity"

    if not opportunity_id:
        return jsonify({"error": "Opportunity ID is required"}), 400

    response = requests.post(GRANTS_API_URL, json={"opportunityId": opportunity_id})

    if response.status_code != 200:
        return jsonify({"error": "Failed to fetch grant data"}), 500

    grant_data = response.json().get("data")

    if not grant_data:
        return jsonify({"error": "Invalid grant data received"}), 500

    # ✅ Return full grant details instead of just storing metadata
    return jsonify({
        "opportunity_id": grant_data["id"],
        "opportunity_number": grant_data["opportunityNumber"],
        "title": grant_data["opportunityTitle"],
        "funding": grant_data.get("estimatedFundingFormatted", "N/A"),
        "eligibility": grant_data.get("applicantEligibilityDesc", "N/A"),
        "deadline": grant_data.get("estApplicationResponseDateStr", "N/A"),
        "agency": grant_data["agencyDetails"]["agencyName"],
        "description": grant_data.get("forecast", {}).get("forecastDesc", "No description available"),
        "requirements": grant_data.get("forecast", {}).get("applicantEligibilityDesc", "Not specified"),
        "objectives": grant_data.get("forecast", {}).get("fundingActivityCategories", "No objectives listed"),
        "evaluation_criteria": grant_data.get("forecast", {}).get("synopsisModifiedFields", "Not specified"),
        "past_grants": []  # Placeholder for future past grant references
    })


API_BASE_URL = "http://127.0.0.1:5000"

@app.route("/generate_grant_proposal", methods=["POST"])
def generate_grant_proposal():
    """Fetch grant details from API, retrieve relevant context, and generate a grant proposal."""
    data = request.json
    opportunity_id = data.get("opportunityId")

    if not opportunity_id:
        return jsonify({"error": "Opportunity ID is required"}), 400

    # ✅ Fetch grant details from the Grants.gov API
    grant_response = requests.post(f"{API_BASE_URL}/fetch_and_store_grant", json={"opportunityId": opportunity_id})

    if grant_response.status_code != 200:
        return jsonify({"error": "Failed to fetch grant details"}), 500

    grant_details = grant_response.json()

    if "error" in grant_details:
        return jsonify(grant_details), 500

    # ✅ Call the `retrieve_relevant_grants` API
    retrieval_response = requests.post(f"{API_BASE_URL}/retrieve_relevant_grants", json={"query": grant_details["title"], "top_n": 5})

    if retrieval_response.status_code != 200:
        return jsonify({"error": "Failed to retrieve relevant grant knowledge"}), 500

    relevant_chunks = retrieval_response.json().get("top_matches", [])

    # ✅ Prepare the structured prompt for GPT-4
    prompt = f"""
### **Grant Proposal Generation**

We are applying for the following grant opportunity. Please generate a professional, structured grant proposal based on the details below.

### **Grant Details:**
- **Title:** {grant_details["title"]}
- **Funding Amount:** {grant_details["funding"]}
- **Deadline:** {grant_details["deadline"]}
- **Eligibility:** {grant_details["eligibility"]}
- **Agency:** {grant_details["agency"]}

### **Grant Description:**
{grant_details["description"]}

### **Grant Requirements:**
{grant_details["requirements"]}

### **Objectives:**
{grant_details["objectives"]}

### **Evaluation Criteria:**
{grant_details["evaluation_criteria"]}

### **Relevant Background Information (from RAG pipeline):**
"""
    for idx, chunk in enumerate(relevant_chunks, start=1):
        prompt += f"\n{idx}. **{chunk['title']}** - {chunk['content'][:500]}...\n"

    prompt += """
---
### **Instructions:**
- Ensure the grant proposal follows best practices.
- The response should be **formal, structured, and professional**.
- Use the provided details and reference materials as **supporting evidence**.
- Structure the response into:
  1. **Introduction**
  2. **Problem Statement**
  3. **Proposed Solution**
  4. **Implementation Plan**
  5. **Expected Outcomes**
  6. **Budget Justification**
  7. **Conclusion**
"""

    # ✅ Call OpenAI GPT-4 to generate the proposal
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional grant writer."},
            {"role": "user", "content": prompt}
        ]
    )

    return jsonify({
        "opportunity_id": opportunity_id,
        "grant_proposal": response.choices[0].message.content.strip(),
        "source_chunks": relevant_chunks
    })