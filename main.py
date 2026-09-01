"""
FastAPI Server & Cloud Endpoint for RAG-Lakehouse Platform
Author: Raja Chakraborty
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from rag_engine import RAGEngine
from gcp_router import route_prompt_to_gcp

app = FastAPI(
    title="RAG-Lakehouse Cloud API",
    description="Serverless API endpoint for RAG retrieval and GCP Gemini model routing.",
    version="1.0.0"
)

# Enable CORS for GitHub Pages UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = RAGEngine()


class QueryRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 3
    category: Optional[str] = None


@app.get("/")
def root():
    return {
        "status": "online",
        "platform": "RAG-Lakehouse on GCP Cloud Run v2",
        "docs": "/docs"
    }


@app.post("/query")
def query_rag_endpoint(req: QueryRequest):
    result = rag_engine.query_rag(query=req.prompt, top_k=req.top_k, category_filter=req.category)
    return result


@app.post("/sanitize")
def sanitize_endpoint(req: QueryRequest):
    result = route_prompt_to_gcp(prompt=req.prompt)
    return result
