from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    text_data: str