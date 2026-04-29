import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

# CHANGED: New index name to force a fresh creation
INDEX_NAME = "pettah-transit-v2" 

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

def get_vectorstore():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # Check if index exists
    if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creating new index: {INDEX_NAME}...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=3072, # Confirmed 3072 from your terminal error
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        
        # NEW: Wait for the index to be fully initialized
        while not pc.describe_index(INDEX_NAME).status['ready']:
            print("Waiting for Pinecone index to be ready...")
            time.sleep(2)
        print("Index is ready!")

    index = pc.Index(INDEX_NAME)
    return PineconeVectorStore(index=index, embedding=get_embeddings())