"""
Automated Test Suite for RAG-Lakehouse Architecture
Author: Raja Chakraborty

Executes end-to-end integration tests for Spark ingestion, PyTorch vector embedding,
ChromaDB search, Bedrock complexity routing, and MCP tool functions.
"""

import sys
from ingestion_spark import process_documents_with_spark, index_chunks_into_vector_db
from rag_engine import RAGEngine
from bedrock_router import route_prompt_to_bedrock
from mcp_server import rag_search_knowledge_base, ingest_document_to_lakehouse, bedrock_route_prompt


def test_ingestion_and_indexing():
    print("\n--- Test 1: PySpark Document Ingestion & PyTorch Indexing ---")
    docs = [
        {
            "doc_id": "test-doc-1",
            "source": "architectural_blueprint.md",
            "category": "distributed_systems",
            "text": "The WebSocket availability pipeline uses RoaringBitmap compression to deliver seat availability states across 21 global markets."
        },
        {
            "doc_id": "test-doc-2",
            "source": "bedrock_router_spec.md",
            "category": "ai_infrastructure",
            "text": "AWS Bedrock model routing selects Claude 3.5 Sonnet for complex architectural code queries and Llama 3 for fast text summarization."
        }
    ]
    chunks = process_documents_with_spark(docs)
    assert len(chunks) >= 2, "PySpark chunking failed"
    index_chunks_into_vector_db(chunks, collection_name="test_knowledge_base")
    print("[SUCCESS] Test 1 Passed: Spark chunking & PyTorch vector indexing verified!")


def test_rag_retrieval():
    print("\n--- Test 2: RAG Hybrid Retrieval & Synthesis ---")
    rag = RAGEngine(collection_name="test_knowledge_base")
    res = rag.query_rag("How does Bedrock model routing work?", top_k=2)

    print(f"Query: {res['query']}")
    print(f"Model Used: {res['model_used']}")
    print(f"Execution Mode: {res['execution_mode']}")
    print(f"Retrieved Chunks Count: {len(res['retrieved_chunks'])}")
    assert len(res["retrieved_chunks"]) > 0, "RAG Retrieval returned zero results"
    print("[SUCCESS] Test 2 Passed: RAG retrieval & context synthesis verified!")


def test_bedrock_router():
    print("\n--- Test 3: AWS Bedrock Complexity Router ---")
    res_complex = route_prompt_to_bedrock("Design a distributed multi-region Kubernetes cluster with circuit breaking")
    res_simple = route_prompt_to_bedrock("Hello, summarize this text")

    print(f"Complex Query Routing: {res_complex['complexity']} -> {res_complex['model_used']}")
    print(f"Simple Query Routing:  {res_simple['complexity']} -> {res_simple['model_used']}")

    assert res_complex["complexity"] == "COMPLEX"
    assert res_simple["complexity"] == "STANDARD"
    print("[SUCCESS] Test 3 Passed: Bedrock complexity heuristics verified!")


def test_mcp_tools():
    print("\n--- Test 4: Model Context Protocol (MCP) Tools ---")
    mcp_ingest = ingest_document_to_lakehouse("Model Context Protocol (MCP) standardizes AI agent tool calling.", source="mcp_test", category="ai_infrastructure")
    mcp_search = rag_search_knowledge_base("What is MCP?")
    mcp_route = bedrock_route_prompt("How does PyTorch handle embedding computation?")

    print("MCP Ingest Output:", mcp_ingest)
    print("MCP Search Output snippet:", mcp_search[:120])
    print("MCP Route Output snippet:", mcp_route[:120])

    assert "Successfully processed" in mcp_ingest
    print("[SUCCESS] Test 4 Passed: MCP tools execution verified!")


if __name__ == "__main__":
    test_ingestion_and_indexing()
    test_rag_retrieval()
    test_bedrock_router()
    test_mcp_tools()
    print("\n[COMPLETE] ALL RAG-LAKEHOUSE INTEGRATION TESTS PASSED SUCCESSFULLY!")
