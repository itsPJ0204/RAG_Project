import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
import chromadb
import uuid
from typing import List, Any
# This matches your requirements.txt
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# --- 1. LOAD ENVIRONMENT ---
print("--- [INGESTION] LOADING ENVIRONMENT ---")
load_dotenv()
# No API keys are needed for local ingestion


# --- 2. PDF PROCESSING ---
def process_all_pdfs(pdf_directory: str) -> List[Any]:
    """Finds all PDFs in a directory and loads them into memory."""
    all_documents = []
    pdf_dir = Path(pdf_directory)

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    print(f"--- [INGESTION] Found {len(pdf_files)} PDF files in {pdf_directory} ---")

    for pdf_file in pdf_files:
        print(f"--- [INGESTION] Processing: {pdf_file.name} ---")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            for doc in documents:
                # Add metadata to each document chunk
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = "pdf"
            all_documents.extend(documents)
            print(f"--- [INGESTION] Loaded {len(documents)} pages from {pdf_file.name} ---")
        except Exception as e:
            print(f"--- [INGESTION] Error loading {pdf_file}: {e} ---")
            
    print(f"\n--- [INGESTION] Total documents loaded: {len(all_documents)} ---")
    return all_documents

# --- 3. TEXT SPLITTING ---
def split_documents(documents: List[Any], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Any]:
    """Splits loaded documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""] # Standard list of separators
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"--- [INGESTION] Split {len(documents)} documents into {len(split_docs)} chunks ---")
    return split_docs

# --- 4. VECTOR STORE MANAGEMENT ---
class VectorStore:
    """Handles loading the embedding model and managing the ChromaDB collection."""
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "data/vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.collection = None
        
        # Initialize the local SentenceTransformer model
        try:
            print("--- [INGESTION] Loading local SentenceTransformer Model ---")
            # This model name matches the one in requirements.txt
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            # Run on CPU for ingestion. You can change to 'cuda' if you have a GPU.
            model_kwargs = {'device': 'cpu'} 
            
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs
            )
            print(f"--- [INGESTION] Local Model '{model_name}' Loaded ---")
        except Exception as e:
            print(f"--- [INGESTION] Error loading embedding model: {e} ---")
            raise

        self._initialize_store()

    def _initialize_store(self):
        """Initializes the persistent ChromaDB client and resets the collection."""
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory) 
            
            # Reset the collection every time we ingest new data
            try:
                self.client.delete_collection(name=self.collection_name)
                print(f"--- [INGESTION] Deleted existing collection: {self.collection_name} ---")
            except Exception as e:
                # This is normal if the collection doesn't exist yet
                print(f"--- [INGESTION] No existing collection to delete, creating new one. ---")

            # Create the new collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"--- [INGESTION] Vector store initialized. Collection: {self.collection_name} ---")

        except Exception as e:
            print(f"--- [INGESTION] Error Initializing vector store: {e} ---")
            raise

    def add_documents(self, documents: List[Any]):
        """Embeds documents in batches and adds them to ChromaDB."""
        print(f"--- [INGESTION] Adding {len(documents)} documents to vector store... ---")
        
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [f"doc_{uuid.uuid4().hex}" for _ in texts]
        
        # Process in batches to manage memory
        batch_size = 100 
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            try:
                print(f"--- [INGESTION] Generating embeddings for batch {i//batch_size + 1} of {len(texts)//batch_size + 1} ---")
                # Generate embeddings locally for this batch
                batch_embeddings = self.embedding_model.embed_documents(batch_texts)
                
                print(f"--- [INGESTION] Adding batch {i//batch_size + 1} to Chroma ---")
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_texts
                )
                print(f"--- [INGESTION] Batch {i//batch_size + 1} successful. ---")
                
            except Exception as e:
                print(f"--- [INGESTION] Error processing batch {i//batch_size + 1}: {e} ---")
        
        print(f"--- [INGESTION] Successfully added documents to vector store. ---")
        print(f"--- [INGESTION] Total documents in collection: {self.collection.count()} ---")


# --- THIS IS THE MAIN EXECUTION BLOCK THAT RUNS THE SCRIPT ---
# --- THIS CODE ONLY RUNS WHEN YOU TYPE "python ingest.py" ---
if __name__ == "__main__":
    print("--- [START] DATA INGESTION ---")
    
    # 1. Load PDFs from the 'data/pdf' folder
    all_pdf_documents = process_all_pdfs("data/pdf")
    
    if all_pdf_documents:
        # 2. Split into chunks
        chunks = split_documents(all_pdf_documents)
        
        # 3. Initialize the VectorStore (which also initializes the embedding model)
        vectorstore = VectorStore(persist_directory="data/vector_store")
        
        # 4. Add documents (embedding is handled inside this method)
        vectorstore.add_documents(chunks)
        
        print("--- [SUCCESS] DATA INGESTION COMPLETE ---")
    else:
        print("--- [SKIP] NO PDFS FOUND IN 'data/pdf'. SKIPPING INGESTION. ---")

