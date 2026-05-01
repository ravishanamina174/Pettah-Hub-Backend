from fastapi import FastAPI, HTTPException
from models import QueryRequest, IngestRequest
from agent import rag_app
from vectorstore import get_vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(
    title="Pettah CTB Bus Stand API",
    description="RAG backend for passenger queries and admin data ingestion.",
    version="1.0.0"
)

# Allow your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://pettah-hub-backend.vercel.app"], # In production, replace "*" with your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/query")
async def ask_passenger_question(req: QueryRequest):
    """Passenger endpoint: Takes a question, runs LangGraph RAG, returns answer."""
    try:
        # Invoke the LangGraph compiled workflow
        result = rag_app.invoke({"question": req.question})
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI Processing Error: {str(e)}")

@app.post("/api/admin/ingest")
async def admin_ingest_data(req: IngestRequest):
    """Admin endpoint: Chunks raw text and uploads embeddings to Pinecone."""
    try:
        # Split large text into readable chunks for the AI
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, 
            chunk_overlap=50
        )
        chunks = text_splitter.split_text(req.text_data)
        
        # Save to Pinecone
        vectorstore = get_vectorstore()
        vectorstore.add_texts(texts=chunks)
        
        return {
            "status": "success", 
            "message": f"Successfully ingested {len(chunks)} chunks into the Pinecone database."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database Ingestion Error: {str(e)}")