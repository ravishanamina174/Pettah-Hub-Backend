# Pettah RAG Backend

A FastAPI-based RAG (Retrieval-Augmented Generation) backend for the Pettah CTB Bus Stand passenger queries and admin data ingestion.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env` and fill in your API keys.
4. Run the app: `uvicorn main:app --reload`

## API Endpoints

- `POST /api/query`: For passenger questions.
- `POST /api/admin/ingest`: For admin data ingestion.

## Deployment on Render

1. Push this code to a GitHub repository.
2. Go to Render.com and create a new Web Service.
3. Connect your GitHub repo.
4. Set the following:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in Render dashboard:
   - `PINECONE_API_KEY`
   - `GOOGLE_API_KEY`
6. Deploy.

Note: Update CORS origins in `main.py` to your frontend URL for production security.