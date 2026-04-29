from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from vectorstore import get_vectorstore

# 1. Define State
class AgentState(TypedDict):
    question: str
    context: str
    answer: str

# 2. System Prompt Architecture
SYSTEM_PROMPT = """You are an official AI assistant for the Pettah CTB Bus Stand (Pettah Transit Hub) in Colombo.
Your absolute priority is accuracy. Answer the passenger's question strictly using the provided context.

Context:
{context}

RULES (CRITICAL):
1. If the context does NOT contain the exact information needed to answer the question, you MUST reply ONLY with: "I don’t have enough information to answer that." Do not guess or make up routes.
2. Format your response strictly as EITHER a single continuous paragraph OR a point-by-point list (line separated). Do not use markdown bolding, tables, or complex structures.

Question: {question}
Answer:"""

# 3. LangGraph Nodes
def retrieve_node(state: AgentState):
    """Retrieves top 3 most relevant chunks from Pinecone."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context}

def generate_node(state: AgentState):
    """Generates the response using Gemini 1.5 Flash."""
    # Using temperature 0 to ensure factual, deterministic responses
    llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", # Or "gemini-2.5-flash" if you prefer stable v2.5 and gemini-3-flash-preview for latest features
    temperature=0.7
)
    prompt = PromptTemplate(template=SYSTEM_PROMPT, input_variables=["context", "question"])
    
    chain = prompt | llm
    response = chain.invoke({"context": state["context"], "question": state["question"]})
    return {"answer": response.content}

# 4. Build the Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile into a runnable application
rag_app = workflow.compile()