
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np
from typing import List, Dict, Any

# --- 1. INITIALIZE APP & LOAD ENVIRONMENT ---
# Load environment variables from .env (for local) or Render dashboard (for production)
load_dotenv()

app = Flask(__name__)
# Allow your React app (from any origin) to call this API
CORS(app) 

# --- 2. LOAD MODELS & DB (GLOBAL - Do this ONCE at startup) ---

print("--- [STARTUP] LOADING MODELS AND DB ---")

# Load Gemini LLM
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env or Render dashboard.")
    llm_client = genai.Client(api_key=gemini_api_key)
    print("Gemini client loaded.")
except Exception as e:
    print(f"Error loading Gemini client: {e}")
    llm_client = None

# Load Embedding Model
try:
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Embedding model (all-MiniLM-L6-v2) loaded.")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedding_model = None

# Connect to the EXISTING Vector Store (built by ingest.py)
try:
    # IMPORTANT: The path must match the 'persist_directory' in ingest.py
    client_db = chromadb.PersistentClient(path="data/vector_store")
    collection = client_db.get_collection(name="pdf_documents")
    print(f"Connected to vector store. Documents in collection: {collection.count()}")
except Exception as e:
    print(f"Error connecting to vector store: {e}")
    collection = None

print("--- [STARTUP] SERVER IS READY ---")

# --- 3. DEFINE THE RAG LOGIC (Helper Functions) ---

def retrieve_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieves context from the vector store."""
    if collection is None or embedding_model is None:
        print("Error: Collection or embedding model not loaded.")
        return []
    
    try:
        query_embedding = embedding_model.encode([query])[0]
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity_score': 1 - dist
                })
        print(f"Retrieved {len(retrieved_docs)} documents for query.")
        return retrieved_docs
    except Exception as e:
        print(f"Error during retrieval: {e}")
        return []

def generate_rag_response(query: str) -> str:
    """Generates a RAG response using the retrieved context and LLM."""
    if llm_client is None:
        return "Error: LLM client is not initialized."

    # 1. Retrieve context
    results = retrieve_context(query)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    # 2. Handle no context
    if not context:
        print("No relevant context found for query.")
        return "No relevant context was found in the documents to answer your question."

    # 3. Build prompt
    prompt = f"""Use the following context to answer the question in brief but not too short.
    Context:
    {context}

    Question: {query}

    Answer:"""

    # 4. Generate response
    try:
        response = llm_client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Error generating response from the AI model."

# --- 4. DEFINE THE API ENDPOINT ---

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    """
    This is the main API endpoint that your React app will call.
    It expects a JSON payload with a "query" key.
    """
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        print(f"--- [REQUEST] Received query: {query[:50]}... ---")
        
        # 5. Get the RAG response
        response_text = generate_rag_response(query)
        
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
