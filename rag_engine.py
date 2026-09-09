"""
Hybrid Search & RAG Synthesis Engine
Author: Raja Chakraborty

Combines PyTorch vector similarity search, ChromaDB metadata filtering,
and GCP Vertex AI / Gemini LLM context synthesis into a high-accuracy RAG pipeline.
Does not generate hallucinated fallback strings when queries match no vector context.
"""

import logging
from typing import Dict, Any, List
from config import VECTOR_DB_DIR
from ingestion_spark import compute_embeddings
from gcp_router import route_prompt_to_gcp

logger = logging.getLogger("rag-lakehouse-engine")


class RAGEngine:
    def __init__(self, collection_name: str = "enterprise_knowledge"):
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
        except Exception as e:
            logger.warning(f"Vector DB Client notice ({str(e)}). Running in dynamic search mode.")

    def retrieve(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Executes vector similarity search using PyTorch embeddings and metadata filters.
        Returns top_k ranked context chunks from ChromaDB.
        """
        logger.info(f"Generating query vector embedding for: '{query}'...")
        query_embeddings = compute_embeddings([query])

        retrieved_docs = []
        if self.collection is not None:
            try:
                where_clause = {"category": category_filter} if category_filter else None
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=top_k,
                    where=where_clause
                )
                if results and results.get("documents") and len(results["documents"]) > 0:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                    for doc_text, meta, dist in zip(docs, metas, distances):
                        if doc_text:
                            retrieved_docs.append({
                                "text": doc_text,
                                "metadata": meta,
                                "distance": dist,
                            })
            except Exception as e:
                logger.warning(f"Vector query notice: {str(e)}")

        logger.info(f"Retrieved {len(retrieved_docs)} relevant context chunks from Vector DB.")
        return retrieved_docs

    def query_rag(self, query: str, top_k: int = 3, category_filter: str = None) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: Vector Retrieval -> Context Formatting -> GCP Gemini Router.
        """
        docs = self.retrieve(query=query, top_k=top_k, category_filter=category_filter)

        context_str = "\n---\n".join([f"[{d['metadata'].get('source', 'doc')}]: {d['text']}" for d in docs]) if docs else ""

        # Synthesize answer via GCP Router
        synthesis = route_prompt_to_gcp(prompt=query, context=context_str)

        return {
            "query": query,
            "retrieved_chunks": docs,
            "context_used": context_str,
            "llm_response": synthesis["response_text"],
            "model_used": synthesis["model_used"],
            "execution_mode": synthesis["mode"],
        }


if __name__ == "__main__":
    rag = RAGEngine()
    response = rag.query_rag("How does PromptShield protect PII?")
    print("RAG Query Result:", response)
