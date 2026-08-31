"""
Model Context Protocol (MCP) Server for RAG-Lakehouse Engine
Author: Raja Chakraborty

Exposes PySpark ingestion, PyTorch Vector Search, and GCP Gemini model routing
as standard MCP tools for AI Agents (Claude Desktop, Cursor, Custom Agents).
"""

import json
import logging
from mcp.server import MCPServer
from ingestion_spark import process_documents_with_spark, index_chunks_into_vector_db
from rag_engine import RAGEngine
from gcp_router import route_prompt_to_gcp

logger = logging.getLogger("rag-lakehouse-mcp")

# Initialize MCP Server
server = MCPServer("RAG-Lakehouse-Engine")
rag_instance = RAGEngine()


@server.tool()
def rag_search_knowledge_base(query: str, category_filter: str = None) -> str:
    """
    Executes RAG retrieval over PyTorch Vector Store and synthesizes an answer using GCP Gemini model routing.
    """
    result = rag_instance.query_rag(query=query, category_filter=category_filter)
    return json.dumps(result, indent=2)


@server.tool()
def ingest_document_to_lakehouse(text: str, source: str = "agent_upload", category: str = "general") -> str:
    """
    Ingests document text through PySpark DataFrame chunking and PyTorch embedding generation into Vector DB.
    """
    docs = [{"doc_id": f"mcp-{hash(text)}", "source": source, "category": category, "text": text}]
    chunks = process_documents_with_spark(docs)
    index_chunks_into_vector_db(chunks)
    return f"Successfully processed {len(chunks)} chunks into Vector Lakehouse."


@server.tool()
def gcp_route_prompt(prompt: str) -> str:
    """
    Evaluates prompt complexity and routes directly to the optimal GCP Gemini LLM model.
    """
    result = route_prompt_to_gcp(prompt=prompt)
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    logger.info("Starting RAG-Lakehouse MCP Tool Server over stdio...")
    server.run(transport="stdio")
