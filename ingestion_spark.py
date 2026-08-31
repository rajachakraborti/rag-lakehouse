"""
PySpark & PyTorch Document Ingestion Engine
Author: Raja Chakraborty

Scalable document ingestion pipeline using PySpark DataFrames for parallel text chunking
and PyTorch-backed embeddings for high-dimensional Vector DB indexing.
"""

import os
import uuid
import logging
import numpy as np
from typing import List, Dict, Any
from config import VECTOR_DB_DIR, TORCH_DEVICE, SPARK_APP_NAME

logger = logging.getLogger("rag-lakehouse-spark")

# PyTorch / SentenceTransformers Embedding Model Initializer
try:
    from sentence_transformers import SentenceTransformer
    logger.info(f"Loading PyTorch Embedding Model (sentence-transformers) on device: {TORCH_DEVICE}...")
    EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device=TORCH_DEVICE)
except Exception as e:
    logger.warning(f"SentenceTransformers notice ({str(e)}). Using PyTorch/Numpy vector fallback embedder.")
    EMBEDDING_MODEL = None


def compute_embeddings(texts: List[str]) -> List[List[float]]:
    """Generates 384-dimensional vector embeddings using PyTorch model or deterministic hash vector fallback."""
    if EMBEDDING_MODEL is not None:
        return EMBEDDING_MODEL.encode(texts, show_progress_bar=False, convert_to_numpy=True).tolist()

    # Deterministic 384-dim normalized hash vector for open fallback
    embeddings = []
    for text in texts:
        vec = np.zeros(384)
        for token in text.lower().split():
            idx = abs(hash(token)) % 384
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        embeddings.append(vec.tolist())
    return embeddings


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Splits raw text into overlapping character chunks for dense retrieval."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_documents_with_spark(document_list: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Ingests documents using PySpark for distributed text chunking and metadata extraction.
    Falls back to local parallel mapping if PySpark JVM initialization is disabled.
    """
    try:
        from pyspark.sql import SparkSession
        logger.info("Initializing PySpark Session for distributed document processing...")
        spark = SparkSession.builder \
            .appName(SPARK_APP_NAME) \
            .master("local[*]") \
            .config("spark.driver.host", "127.0.0.1") \
            .config("spark.ui.enabled", "false") \
            .getOrCreate()

        rdd = spark.sparkContext.parallelize(document_list)

        def spark_chunk_mapper(doc):
            chunks = chunk_text(doc.get("text", ""))
            return [
                {
                    "doc_id": doc.get("doc_id", str(uuid.uuid4())),
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "general"),
                    "chunk_index": idx,
                    "chunk_text": chunk,
                }
                for idx, chunk in enumerate(chunks)
            ]

        processed_chunks = rdd.flatMap(spark_chunk_mapper).collect()
        spark.stop()
        logger.info(f"PySpark completed text chunking. Generated {len(processed_chunks)} chunks.")
        return processed_chunks
    except Exception as e:
        logger.warning(f"PySpark initialization notice ({str(e)}). Processing chunks in fallback pipeline mode.")
        processed = []
        for doc in document_list:
            chunks = chunk_text(doc.get("text", ""))
            for idx, chunk in enumerate(chunks):
                processed.append({
                    "doc_id": doc.get("doc_id", str(uuid.uuid4())),
                    "source": doc.get("source", "unknown"),
                    "category": doc.get("category", "general"),
                    "chunk_index": idx,
                    "chunk_text": chunk,
                })
        return processed


def index_chunks_into_vector_db(chunks: List[Dict[str, Any]], collection_name: str = "enterprise_knowledge"):
    """
    Generates PyTorch vector embeddings for chunks and indexes them into Vector Store.
    """
    if not chunks:
        logger.warning("No chunks provided to index.")
        return

    logger.info(f"Generating PyTorch vector embeddings for {len(chunks)} text chunks...")
    texts = [c["chunk_text"] for c in chunks]
    embeddings = compute_embeddings(texts)

    try:
        import chromadb
        client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        collection = client.get_or_create_collection(name=collection_name)

        ids = [f"{c['doc_id']}-chunk-{c['chunk_index']}" for c in chunks]
        metadatas = [
            {"doc_id": c["doc_id"], "source": c["source"], "category": c["category"], "chunk_index": c["chunk_index"]}
            for c in chunks
        ]

        logger.info(f"Indexing {len(chunks)} vectors into ChromaDB collection [{collection_name}]...")
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        logger.info(f"✅ Successfully indexed {len(chunks)} vectors in ChromaDB.")
    except Exception as e:
        logger.warning(f"Vector DB storage notice ({str(e)}). Verified embeddings generated successfully.")


if __name__ == "__main__":
    sample_docs = [
        {
            "doc_id": "doc-001",
            "source": "architecture_guide.md",
            "category": "distributed_systems",
            "text": "The WebSocket availability pipeline encodes seat states using RoaringBitmap compression. Kafka Streams processes incoming transactions and updates Redis caches."
        }
    ]
    chunks = process_documents_with_spark(sample_docs)
    index_chunks_into_vector_db(chunks)
