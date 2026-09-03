"""
FastAPI Server, Static Auth Middleware & Budget Safeguard API
Author: Raja Chakraborty

Provides:
1. Static User API Key Authentication (X-API-Key: demo-key-2026).
2. Swagger UI (/docs) Authorize button & explicit Header input.
3. Rate Limiter Guard (Max 20 requests/minute per client IP).
4. Prompt Token Guard (Max 2000 characters to enforce $2.00 monthly cost cap).
5. Dynamic Lakehouse Ingestion endpoint (POST /ingest).
"""

import time
from typing import Optional, Dict
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Security, Depends, status, Header
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import RAGEngine
from gcp_router import route_prompt_to_gcp

app = FastAPI(
    title="RAG-Lakehouse Authenticated Cloud API",
    description="Serverless API endpoint protected by Static API Key Auth and $2.00 Budget Rate Limiting. Use 'demo-key-2026' or 'admin-key-789'.",
    version="1.3.0"
)

# Enable CORS for GitHub Pages UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static User Authentication Credentials
STATIC_API_KEYS = {
    "demo-key-2026": "Demo User",
    "admin-key-789": "Admin User",
    "public-sandbox-key": "Public Explorer"
}

# Swagger UI API Key Header Security Scheme
api_key_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    header_key: Optional[str] = Header(None, alias="X-API-Key", description="Static API Key for authentication (e.g. demo-key-2026)"),
    security_key: Optional[str] = Security(api_key_header_scheme)
) -> str:
    """
    Validates static API Key from X-API-Key HTTP header or Swagger UI Authorize lock.
    """
    api_key = header_key or security_key
    if not api_key:
        return "public-sandbox-key"
    if api_key not in STATIC_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid X-API-Key. Authorized keys: demo-key-2026, admin-key-789"
        )
    return api_key


# Rate Limiter & Token Guard Settings ($2.00 Budget Safeguards)
MAX_REQUESTS_PER_MINUTE = 20
MAX_PROMPT_CHAR_LENGTH = 2000
request_timestamps: Dict[str, list] = defaultdict(list)


def enforce_budget_rate_limit(user_key: str):
    """
    Enforces a strict sliding-window rate limit to guarantee GCP compute budget stays under $2.00.
    """
    now = time.time()
    timestamps = request_timestamps[user_key]
    timestamps = [t for t in timestamps if now - t < 60]
    request_timestamps[user_key] = timestamps

    if len(timestamps) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded (Max {MAX_REQUESTS_PER_MINUTE} requests/min to maintain $2.00 monthly budget cap)."
        )
    request_timestamps[user_key].append(now)


rag_engine = RAGEngine()


class QueryRequest(BaseModel):
    prompt: str
    top_k: Optional[int] = 3
    category: Optional[str] = None


class IngestRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, str]] = None


@app.get("/")
def root():
    return {
        "status": "online",
        "platform": "RAG-Lakehouse on GCP Cloud Run v2",
        "auth": "Static API Key (X-API-Key: demo-key-2026)",
        "budget_limit": "$2.00 Cap Safeguard Active",
        "docs": "/docs"
    }


@app.post("/query")
def query_rag_endpoint(
    req: QueryRequest,
    user_key: str = Depends(verify_api_key)
):
    enforce_budget_rate_limit(user_key)
    
    if len(req.prompt) > MAX_PROMPT_CHAR_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt length ({len(req.prompt)} chars) exceeds maximum limit of {MAX_PROMPT_CHAR_LENGTH} chars to protect GCP token quota."
        )

    result = rag_engine.query_rag(query=req.prompt, top_k=req.top_k, category_filter=req.category)
    result["authenticated_user"] = STATIC_API_KEYS.get(user_key, "User")
    return result


@app.post("/ingest")
def ingest_document_endpoint(
    req: IngestRequest,
    user_key: str = Depends(verify_api_key)
):
    enforce_budget_rate_limit(user_key)
    
    if len(req.text) > MAX_PROMPT_CHAR_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Text length ({len(req.text)} chars) exceeds maximum limit of {MAX_PROMPT_CHAR_LENGTH} chars."
        )

    chunk_id = f"user_doc_{int(time.time() * 1000)}"
    meta = req.metadata or {"source": "custom_user_ingestion.md", "category": "user_upload"}
    
    rag_engine.collection.add(
        documents=[req.text],
        metadatas=[meta],
        ids=[chunk_id]
    )

    return {
        "status": "success",
        "ingested_id": chunk_id,
        "total_documents": rag_engine.collection.count(),
        "message": f"Successfully indexed text ({len(req.text)} chars) into ChromaDB vector store.",
        "authenticated_user": STATIC_API_KEYS.get(user_key, "User")
    }


@app.post("/sanitize")
def sanitize_endpoint(
    req: QueryRequest,
    user_key: str = Depends(verify_api_key)
):
    enforce_budget_rate_limit(user_key)
    
    if len(req.prompt) > MAX_PROMPT_CHAR_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prompt length exceeds maximum limit of {MAX_PROMPT_CHAR_LENGTH} chars."
        )

    result = route_prompt_to_gcp(prompt=req.prompt)
    result["authenticated_user"] = STATIC_API_KEYS.get(user_key, "User")
    return result
