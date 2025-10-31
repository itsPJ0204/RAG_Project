import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import chromadb
import numpy as np
from typing import List, Dict, Any
# --- 1. INITIALIZE APP & LOAD ENVIRONMENT ---
load_dotenv()
app = Flask(__name__)
CORS(app) 

print("--- [STARTUP] LOADING MODELS AND DB ---")

# --- CHANGED ---
# Load API Key and configure the new SDK
try:
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env or Render dashboard.")
    
    # Configure the Google AI SDK (used by both LLM and Embeddings)
    genai.configure(api_key=gemini_api_key)
    
    # Load Gemini LLM using the new SDK
    # Using 1.5-flash (2.5-flash is not a valid model name)
    llm_model = genai.GenerativeModel("gemini-1.5-flash") 
    print("Gemini model (gemini-1.5-flash) loaded.")

except Exception as e:
    print(f"Error loading Gemini client: {e}")
    llm_model = None

# --- CHANGED ---
# Load Embedding Model (via API, no memory usage)
try:
    # We re-use the configured API key from above
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Google AI Embedding model (API) loaded.")
except Exception as e:
    print(f"Error loading Google AI embedding model: {e}")
    embedding_model = None
# --- END CHANGED ---

# Connect to the EXISTING Vector Store (built by ingest.py)
try:
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
        # --- CHANGED ---
        # Use .embed_query() for single queries (returns List[float])
        query_embedding = embedding_model.embed_query(query)
        
        results = collection.query(
            # The embedding is already a list, no .tolist() needed
            query_embeddings=[query_embedding], 
            n_results=top_k
        )
        # --- END CHANGED ---
        
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
    # --- CHANGED ---
    # Check the new model variable
    if llm_model is None:
        return "Error: LLM client is not initialized."

    results = retrieve_context(query)
    context = "\n\n".join([doc['content'] for doc in results]) if results else ""

    if not context:
        print("No relevant context found for query.")
        return "No relevant context was found in the documents to answer your question."

    prompt = f"""Use the following context to answer the question in brief but not too short.
    Context:
    {context}

    Question: {query}

    Answer:"""

    try:
        # --- CHANGED ---
        # Use the new SDK's generate_content method
        response = llm_model.generate_content(prompt)
        return response.text
        # --- END CHANGED ---
    except Exception as e:
        print(f"Error generating Gemini response: {e}")
        return "Error generating response from the AI model."

# --- 4. DEFINE THE API ENDPOINT ---

@app.route("/api/chat", methods=["POST"])
def chat_handler():
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        print(f"--- [REQUEST] Received query: {query[:50]}... ---")
        
        response_text = generate_rag_response(query)
        
        return jsonify({"response": response_text})
    
    except Exception as e:
        print(f"Error in /api/chat: {e}")
        return jsonify({"error": "An internal server error occurred."}), 500
