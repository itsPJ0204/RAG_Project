import os
from dotenv import load_dotenv 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import numpy as np
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import chromadb
import uuid
from typing import List, Dict, Any, List 

# ### 1. Read all the pdfs inside the directory
def process_all_pdfs(pdf_directory):
    all_documents=[]
    pdf_dir=Path(pdf_directory)

    pdf_files=list(pdf_dir.glob("**/*.pdf"))
    print(f"Found {len(pdf_files)} pdf files")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file}")
        try:
            loader=PyPDFLoader(str(pdf_file))
            documents=loader.load()
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = "pdf"
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} pages")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
    print(f"\n Total Documents Loaded: {len(all_documents)}")
    return all_documents

# ### 2. Text Splitting get into chunks
def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    return split_docs

# ### 3. Embedding Manager
class EmbeddingManager:
    # --- CHANGED ---
    # Default to Google's embedding model
    def __init__(self, model_name: str="models/embedding-001"): 
        self.model_name=model_name
        self.model=None
        self._load_model()

    def _load_model(self):
        try:
            # --- CHANGED ---
            # Load API key and configure the Google AI client
            print(f"Loading embedding model: {self.model_name}")
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY not found. Set it in .env or Render dashboard.")
            
            genai.configure(api_key=api_key)
            
            # Initialize the LangChain embeddings class
            self.model = GoogleGenerativeAIEmbeddings(model=self.model_name)
            print("Google AI Embedding model loaded successfully.")
            # --- END CHANGED ---
        except Exception as e:
            print(f"Error Loading Model: {self.model_name} : {e}")
            raise

    # --- CHANGED ---
    # Updated return type hint
    def generate_embeddings(self, texts: list[str]) -> List[List[float]]: 
        if not self.model:
            raise ValueError("Model Not Loaded") 
        
        print(f"Generating embeddings for {len(texts)} texts via API...")
        # --- CHANGED ---
        # Use embed_documents() which returns List[List[float]]
        embeddings = self.model.embed_documents(texts) 
        print(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings

# ### 4. VectorStore Manager
class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory) 

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error Initializing vector store: {e}")
            raise

    # --- CHANGED ---
    # Updated type hint for embeddings
    def add_documents(self, documents: List[Any], embeddings: List[List[float]]): 
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")

        print(f"Adding {len(documents)} documents to vector store...")

        ids = []
        metadata_list = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            meta = dict(doc.metadata)
            meta['doc_index'] = i
            meta['content_length'] = len(doc.page_content)
            metadata_list.append(meta)

            documents_text.append(doc.page_content)
            
            # --- CHANGED ---
            # The embedding is already a list[float], no .tolist() needed
            embeddings_list.append(embedding) 

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadata_list,
                documents=documents_text
            )

            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

# --- This is the main execution block that runs the script ---
if __name__ == "__main__":
    # --- CHANGED ---
    # Load .env variables (GEMINI_API_KEY) before starting
    print("Loading environment variables...")
    load_dotenv() 
    
    print("--- [START] DATA INGESTION ---")
    
    all_pdf_documents = process_all_pdfs("data/pdf")
    
    if all_pdf_documents:
        chunks = split_documents(all_pdf_documents)
        
        # This will now use the Google API via EmbeddingManager
        embedding_manager = EmbeddingManager() 
        vectorstore = VectorStore(persist_directory="data/vector_store")
        
        texts = [doc.page_content for doc in chunks]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        vectorstore.add_documents(chunks, embeddings)
        
        print("--- [SUCCESS] DATA INGESTION COMPLETE ---")
    else:
        print("--- [SKIP] NO PDFS FOUND IN 'data/pdf'. SKIPPING INGESTION. ---")
