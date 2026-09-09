import math
import re
import hashlib
import logging
from typing import Dict, Any, List
from config import VECTOR_DB_DIR
from ingestion_spark import compute_embeddings
from gcp_router import route_prompt_to_gcp
from checksum_cache import checksum_cache

logger = logging.getLogger("rag-lakehouse-engine")


def compute_bm25_scores(query: str, documents: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """
    Computes Okapi BM25 relevance scores for exact keyword & token matches.
    Provides deterministic lexical matching to complement probabilistic vector search.
    """
    if not documents or not query.strip():
        return [0.0] * len(documents)

    def tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    query_tokens = tokenize(query)
    doc_tokens = [tokenize(d) for d in documents]
    N = len(documents)
    avgdl = sum(len(d) for d in doc_tokens) / max(N, 1)

    df = {}
    for q_token in set(query_tokens):
        df[q_token] = sum(1 for d in doc_tokens if q_token in d)

    scores = []
    for d_tokens in doc_tokens:
        score = 0.0
        doc_len = len(d_tokens)
        term_counts = {}
        for token in d_tokens:
            term_counts[token] = term_counts.get(token, 0) + 1

        for q_token in query_tokens:
            if q_token in term_counts:
                tf = term_counts[q_token]
                n_q = df.get(q_token, 0)
                idf = math.log((N - n_q + 0.5) / (n_q + 0.5) + 1.0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_len / max(avgdl, 1)))
                score += idf * (numerator / max(denominator, 1e-6))
        scores.append(score)
    return scores


def reciprocal_rank_fusion(vector_docs: List[Dict[str, Any]], lexical_docs: List[Dict[str, Any]], rrf_k: int = 60) -> List[Dict[str, Any]]:
    """
    Combines Vector Search (semantic) and BM25 Lexical Search (exact keyword) using Reciprocal Rank Fusion.
    RRF_Score = 1/(k + Rank_vector) + 1/(k + Rank_lexical)
    """
    rrf_scores = {}
    doc_map = {}

    for rank, doc in enumerate(vector_docs):
        h = hashlib.sha256(doc["text"].strip().encode("utf-8")).hexdigest()
        doc_map[h] = doc
        rrf_scores[h] = rrf_scores.get(h, 0.0) + (1.0 / (rrf_k + rank + 1))

    for rank, doc in enumerate(lexical_docs):
        h = hashlib.sha256(doc["text"].strip().encode("utf-8")).hexdigest()
        if h not in doc_map:
            doc_map[h] = doc
        rrf_scores[h] = rrf_scores.get(h, 0.0) + (1.0 / (rrf_k + rank + 1))

    sorted_hashes = sorted(rrf_scores.keys(), key=lambda h: rrf_scores[h], reverse=True)
    fused_docs = []
    for h in sorted_hashes:
        doc = doc_map[h].copy()
        doc["rrf_score"] = rrf_scores[h]
        doc["search_type"] = "HYBRID_BM25_VECTOR"
        fused_docs.append(doc)
    return fused_docs


class RAGEngine:
    def __init__(self, collection_name: str = "enterprise_knowledge"):
        self.collection_name = collection_name
        self.chroma_client = None
        self.collection = None
        self.checksum_cache = checksum_cache

        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
            self.collection = self.chroma_client.get_or_create_collection(name=collection_name)
            self.sync_checksum_registry()
        except Exception as e:
            logger.warning(f"Vector DB Client notice ({str(e)}). Running in dynamic search mode.")

    def sync_checksum_registry(self):
        """Pre-seeds distributed checksum cache from Vector DB collection on cold start."""
        if self.collection is None:
            return
        try:
            res = self.collection.get()
            docs = res.get("documents") or []
            for doc_text in docs:
                if doc_text:
                    self.checksum_cache.add(doc_text)
            logger.info("Distributed Checksum Cache synchronized with ChromaDB collection.")
        except Exception as e:
            logger.warning(f"Checksum cache sync notice ({str(e)}).")

    def is_duplicate_payload(self, text: str, custom_key: str = None) -> bool:
        """Fast multi-layer check using Distributed Checksum Cache."""
        return self.checksum_cache.is_duplicate(text, custom_key)

    def register_checksum(self, text: str, custom_key: str = None) -> str:
        """Registers newly indexed payload in Distributed Checksum Cache."""
        return self.checksum_cache.add(text, custom_key)

    def retrieve(self, query: str, top_k: int = 3, category_filter: str = None) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search: PyTorch Dense Vector Search + BM25 Lexical Keyword Search.
        Fuses ranked candidate lists via Reciprocal Rank Fusion (RRF) and deduplicates outputs.
        """
        logger.info(f"Executing Hybrid Retrieval (Vector + BM25) for query: '{query}'...")
        query_embeddings = compute_embeddings([query])

        vector_retrieved = []
        all_corpus_docs = []
        all_corpus_metas = []

        if self.collection is not None:
            try:
                where_clause = {"category": category_filter} if category_filter else None

                # 1. Vector Search Candidate Retrieval
                results = self.collection.query(
                    query_embeddings=query_embeddings,
                    n_results=min(top_k * 3, max(self.collection.count(), 1)),
                    where=where_clause
                )
                if results and results.get("documents") and len(results["documents"]) > 0:
                    docs = results["documents"][0]
                    metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
                    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

                    for doc_text, meta, dist in zip(docs, metas, distances):
                        if doc_text:
                            vector_retrieved.append({
                                "text": doc_text,
                                "metadata": meta,
                                "distance": dist,
                            })

                # 2. Fetch corpus documents for BM25 Lexical Ranking
                all_res = self.collection.get()
                all_corpus_docs = all_res.get("documents") or []
                all_corpus_metas = all_res.get("metadatas") or [{}] * len(all_corpus_docs)
            except Exception as e:
                logger.warning(f"Retrieval notice: {str(e)}")

        # 3. Compute BM25 Lexical Scores across collection corpus
        bm25_scores = compute_bm25_scores(query, all_corpus_docs)
        lexical_tuples = sorted(
            zip(bm25_scores, all_corpus_docs, all_corpus_metas),
            key=lambda x: x[0],
            reverse=True
        )

        lexical_retrieved = []
        for score, doc_text, meta in lexical_tuples:
            if score > 0.0 and doc_text:
                lexical_retrieved.append({
                    "text": doc_text,
                    "metadata": meta,
                    "bm25_score": score
                })
            if len(lexical_retrieved) >= top_k * 3:
                break

        # 4. Fuse Vector & BM25 Results via Reciprocal Rank Fusion (RRF)
        fused_results = reciprocal_rank_fusion(vector_retrieved, lexical_retrieved)

        logger.info(f"Hybrid Search fused {len(vector_retrieved)} vector candidates & {len(lexical_retrieved)} BM25 candidates into {len(fused_results)} top-ranked results.")
        return fused_results[:top_k]

    def query_rag(self, query: str, top_k: int = 3, category_filter: str = None) -> Dict[str, Any]:
        """
        End-to-end RAG pipeline: Hybrid Retrieval -> Context Formatting -> GCP Gemini Router.
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
    response = rag.query_rag("IEEE Senior Member")
    print("Hybrid RAG Query Result:", response)
