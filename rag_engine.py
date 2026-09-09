"""
Hybrid Search & RAG Synthesis Engine
Author: Raja Chakraborty

Combines PyTorch vector similarity search, ChromaDB metadata filtering,
and GCP Vertex AI / Gemini LLM context synthesis into a high-accuracy RAG pipeline.
Does not generate hallucinated fallback strings when queries match no vector context.
"""

import hashlib
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
        self.checksum_registry = set()

        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            self.sync_checksum_registry()
        except Exception as e:
            logger.warning(f"Vector DB Client notice ({str(e)}). Running in dynamic search mode.")

    def sync_checksum_registry(self):
        """Pre-seeds high-level in-memory checksum registry from Vector DB for O(1) quick rejection."""
        if self.collection is None:
            return
        try:
            res = self.collection.get()
            docs = res.get("documents") or []
            ids = res.get("ids") or []
            for doc_id, doc_text in zip(ids, docs):
                if doc_id.startswith("doc_"):
                    self.checksum_registry.add(doc_id.replace("doc_", ""))
                if doc_text:
                    h = hashlib.sha256(doc_text.strip().encode("utf-8")).hexdigest()[:16]
                    self.checksum_registry.add(h)
            logger.info(f"High-Level Checksum Registry initialized with {len(self.checksum_registry)} document hashes.")
        except Exception as e:
            logger.warning(f"Checksum registry sync notice ({str(e)}).")

    def is_duplicate_payload(self, text: str, custom_key: str = None) -> bool:
        """Sub-millisecond O(1) high-level check to quickly reject duplicate ingestion payloads."""
        key = (custom_key or text).strip()
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return h in self.checksum_registry

    def register_checksum(self, text: str, custom_key: str = None) -> str:
        """Registers a newly indexed payload hash in the high-level in-memory registry."""
        key = (custom_key or text).strip()
        h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        self.checksum_registry.add(h)
        return h

    def retrieve(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Executes vector similarity search using PyTorch embeddings and metadata filters.
        Deduplicates retrieved result chunks by content hash before returning top_k unique matches.
        """
        logger.info(f"Generating query vector embedding for: '{query}'...")
        query_embeddings = compute_embeddings([query])

        retrieved_docs = []
        if self.collection is not None:
            try:
                where_clause = {"category": category_filter} if category_filter else None
                # Query extra candidates (top_k * 2) to account for deduplication
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=min(top_k * 2, max(self.collection.count(), 1)),
                    where=where_clause
                )
                if results and results.get("documents") and len(results["documents"]) > 0:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                    seen_hashes = set()
                    for doc_text, meta, dist in zip(docs, metas, distances):
                        if doc_text:
                            content_hash = hashlib.sha256(doc_text.strip().encode("utf-8")).hexdigest()[:16]
                            if content_hash not in seen_hashes:
                                seen_hashes.add(content_hash)
                                retrieved_docs.append({
                                    "text": doc_text,
                                    "metadata": meta,
                                    "distance": dist,
                                })
                            if len(retrieved_docs) >= top_k:
                                break
            except Exception as e:
                logger.warning(f"Vector query notice: {str(e)}")

        logger.info(f"Retrieved {len(retrieved_docs)} unique context chunks from Vector DB.")
        return retrieved_docs[:top_k]

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
