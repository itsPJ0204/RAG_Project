import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from typing import List, Dict, Any

# --- 1. INITIALIZE APP & LOAD ENVIRONMENT ---
load_dotenv()
app = Flask(__name__)

# CORS - Allow all origins (for testing, restrict in production)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# --- 2. LOAD MODELS & DB (GLOBAL - Do this ONCE at startup) ---

print("--- [STARTUP] LOADING MODELS AND DB ---")

# Load Gemini LLM (for generation)
try:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("--- [STARTUP] Gemini client initialized successfully ---")
except Exception as e:
    print(f"--- [STARTUP ERROR] Error loading Gemini client: {e} ---")
    gemini_client = None

# Load LOCAL Embedding Model
print("--- [STARTUP] Loading local SentenceTransformer Model ---")
try:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    print(f"--- [STARTUP] Local Model '{model_name}' loaded successfully ---")
except Exception as e:
    print(f"--- [STARTUP ERROR] Error loading embedding model: {e} ---")
    embedding_model = None

# Connect to Vector Store
try:
    vector_store_path = os.path.join(os.getcwd(), "data", "vector_store")
    print(f"--- [STARTUP] Looking for vector store at: {vector_store_path} ---")
    
    if not os.path.exists(vector_store_path):
        print(f"--- [STARTUP ERROR] Vector store path does not exist! ---")
        collection = None
    else:
        client_db = chromadb.PersistentClient(path=vector_store_path)
        collection = client_db.get_collection(name="pdf_documents")
        print(f"--- [STARTUP] Connected to vector store. Documents: {collection.count()} ---")
except Exception as e:
    print(f"--- [STARTUP ERROR] Error connecting to vector store: {e} ---")
    collection = None

print("--- [STARTUP] SERVER IS READY ---")

# --- 3. HEALTH CHECK ENDPOINT ---
@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint to verify server is running"""
    status = {
        "status": "running",
        "gemini_client": "initialized" if gemini_client else "error",
        "embedding_model": "loaded" if embedding_model else "error",
        "vector_store": f"{collection.count()} documents" if collection else "error"
    }
    return jsonify(status), 200

@app.route("/api/health", methods=["GET"])
def api_health():
    """API health check"""
    return jsonify({"status": "ok"}), 200

# --- 4. RAG LOGIC ---

def retrieve_context(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Retrieves context from the vector store."""
    if collection is None or embedding_model is None:
        print("--- [RETRIEVAL ERROR] Collection or embedding model not loaded ---")
        return []
    
    try:
        print(f"--- [RETRIEVAL] Embedding query: '{query[:50]}...' ---")
        query_embedding = embedding_model.embed_query(query)
        
        print("--- [RETRIEVAL] Querying vector store... ---")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        retrieved_docs = []
        if results and results.get('documents'):
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                retrieved_docs.append({
                    'content': doc,
                    'metadata': meta,
                    'similarity_score': 1 - dist
                })
            print(f"--- [RETRIEVAL] Retrieved {len(retrieved_docs)} documents ---")
        else:
            print("--- [RETRIEVAL] No documents found ---")
            
        return retrieved_docs
    except Exception as e:
        print(f"--- [RETRIEVAL ERROR] {e} ---")
        return []

def generate_rag_response(query: str, chat_history: List[Dict[str, str]]) -> str:
    """Generates a RAG response using retrieved context and LLM."""
    if gemini_client is None:
        print("--- [GENERATE ERROR] Gemini client not initialized ---")
        return "Error: AI model is not initialized. Please check server logs."

    # Retrieve context
    results = retrieve_context(query)
    
    # Format context
    context_str = "\n\n---\n\n".join([
        f"Source: {doc['metadata'].get('source_file', 'N/A')}\nContent: {doc['content']}" 
        for doc in results
    ])

    if not context_str:
        print("--- [GENERATE] No relevant context found ---")
        context_str = "No relevant documents were found."

    # Build chat history context
    history_context = ""
    if chat_history:
        history_context = "\n\n--- PREVIOUS CONVERSATION ---\n"
        for item in chat_history[-6:]:
            sender = item.get("sender", "unknown")
            text = item.get("text", "")
            history_context += f"{sender.upper()}: {text}\n"
        history_context += "--- END PREVIOUS CONVERSATION ---\n\n"
    
    # Build final prompt
    prompt = f"""{history_context}You are an expert assistant for screenwriting and filmmaking.
Your knowledge base consists ONLY of the documents provided below.
Answer the user's query based ONLY on the provided context.
Do not use any outside knowledge. If the answer is not in the context, state that clearly.
Be detailed and helpful.

--- CONTEXT DOCUMENTS ---
{context_str}
--- END CONTEXT ---

User Query: {query}

Answer:"""

    # Generate response
    try:
        print("--- [GENERATE] Sending prompt to Gemini... ---")
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        print("--- [GENERATE] Response received from Gemini ---")
        return response.text
    except Exception as e:
        print(f"--- [GENERATE ERROR] {e} ---")
        return f"Error: Could not generate response. Details: {str(e)}"

# --- 5. API ENDPOINT ---

@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat_handler():
    """Main API endpoint for the React app."""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        return "", 200
    
    try:
        data = request.json
        if not data or "query" not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data["query"]
        chat_history = data.get("history", [])
        
        print(f"--- [REQUEST] Received query: '{query[:50]}...' ---")
        
        # Get RAG response
        response_text = generate_rag_response(query, chat_history)
        
        return jsonify({"response": response_text}), 200
    
    except Exception as e:
        print(f"--- [REQUEST ERROR] {e} ---")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# --- 6. RUN THE APP ---

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"--- [STARTUP] Starting server on port {port} ---")
    app.run(host='0.0.0.0', port=port, debug=False)