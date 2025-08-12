# main.py
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

from file_loader import extracting_text
from rag_pipeline import create_retriever
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set.")

# FastAPI app
app = FastAPI(title="RAG Pipeline API")

# Global retriever & chain (kept in memory)
rag_chain = None

# Request & Response Models
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# ---------------------------
# UPLOAD Endpoint (in-memory)
# ---------------------------
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global rag_chain

    file_bytes = await file.read()

    # Extract directly from bytes
    extracted_docs = extracting_text(file_bytes, file.filename)
    if not extracted_docs:
        raise HTTPException(status_code=400, detail="Failed to extract text from document.")

    hybrid_retriever = create_retriever(extracted_docs)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert assistant. Answer only based on the provided context. "
                   "If you cannot find the answer, say you don't have enough information."),
        ("user", "Context: {context}\n\nQuestion: {input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(hybrid_retriever, combine_docs_chain)

    return {"message": f"Document '{file.filename}' uploaded and retriever created in memory."}

# ---------------------------
# ASK Endpoint
# ---------------------------
@app.post("/ask", response_model=QueryResponse)
async def ask_question(query: QueryRequest):
    """
    Ask a question to the uploaded document.
    """
    global rag_chain
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No document uploaded yet. Please upload a document first.")

    # Run query through RAG
    response = rag_chain.invoke({"input": query.question})
    final_answer = response["answer"]
    retrieved_docs = [doc.page_content[:200] for doc in response["context"]]  # First 200 chars

    return QueryResponse(answer=final_answer)
