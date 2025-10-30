#!/usr/bin/env python
# coding: utf-8

# ### RAG Pipelines- Data Ingestion to Vector DB Pipeline

# In[1]:


import os 
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path


# In[2]:


### Read all the pdfs inside the directory
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
            print(f"Error: {e}")
    print(f"\n Total Documents Loaded: {len(all_documents)}")
    return all_documents

all_pdf_documents=process_all_pdfs("data/pdf")


# In[3]:


all_pdf_documents


# In[4]:


### Text Splitting get into chunks

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    return split_docs


# In[5]:


chunks=split_documents(all_pdf_documents)
chunks


# ### Embedding and VectorStoreDB

# In[6]:


import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity


# In[7]:


class EmbeddingManager:
    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        self.model_name=model_name
        self.model=None
        self._load_model()
    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model=SentenceTransformer(self.model_name)
            print(f"Model Loaded Successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error Loading Model: {self.model_name} : {e}")
            raise
    def generate_embeddings(self,texts: list[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Model Not Loaded") 
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
### Initialize the embedding manager
embedding_manager=EmbeddingManager()
embedding_manager


# In[8]:


import os
import uuid
import numpy as np
from typing import List, Any
import chromadb

class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
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

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
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
            embeddings_list.append(embedding.tolist())

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

vectorstore=VectorStore()
vectorstore


# In[9]:


chunks


# In[10]:


### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

### Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

### Store in the vector db

vectorstore.add_documents(chunks,embeddings)

print("\n=== DEBUG: Checking what’s inside the vector store ===")
all_docs = vectorstore.collection.get()
print(f"Total docs in collection: {len(all_docs['documents'])}")
print("Example document snippet:", all_docs['documents'][0][:200])
print("Metadata:", all_docs['metadatas'][0])


# ### Retriever Pipeline From Vector Store

# In[11]:


class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score Threshold: {score_threshold}")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    similarity_score = 1 - distance
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

                if retrieved_docs:
                    print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
                else:
                    print("No documents found")

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieving: {e}")
            return []


# ✅ Instantiate correctly (outside the class)
rag_retriever = RAGRetriever(vectorstore, embedding_manager)


# In[12]:


rag_retriever


# In[14]:


rag_retriever.retrieve("What is a startup?", score_threshold=0)


# ### Integration VectorDB Context pipeline with LLM Output

# In[18]:


from google import genai
import os 
from dotenv import load_dotenv
load_dotenv()

gemini_api_key="AIzaSyDWWlFGoNySQZELl9nKSSOTYN3GJ0Orca8"

client = genai.Client(api_key=gemini_api_key)

def rag_simple(query, retriever,client,top_k=3):
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found"
    prompt=f"""Use the following context to answer the question in brief but not too short.
    Context:
    {context}

    Question: {query}

    Answer:"""

    response = client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt
)
    return response.text






