"""
Distributed & Persistent Idempotency Checksum Cache Module
Author: Raja Chakraborty

Provides persistent SHA-256 deduplication across container restarts, multi-instance scale-out,
and cold starts on GCP Cloud Run.

Supports:
1. GCP Firestore Key-Value Store (0-Cost Free Tier on GCP)
2. Redis REST / Upstash HTTP API (if REDIS_REST_URL configured)
3. Local SQLite & ChromaDB Cold-Start Auto-Hydration (Zero-config fallback)
"""

import os
import sqlite3
import hashlib
import logging
import urllib.request
import json
from typing import Optional, Dict, Any, Set
from config import VECTOR_DB_DIR

logger = logging.getLogger("rag-lakehouse-cache")


class ChecksumCacheEngine:
    def __init__(self, db_dir: str = VECTOR_DB_DIR):
        self.db_dir = db_dir
        self.sqlite_path = os.path.join(db_dir, "checksum_registry.db")
        self.local_memory_set: Set[str] = set()
        self.firestore_db = None
        self.redis_url = os.getenv("REDIS_REST_URL")
        self.redis_token = os.getenv("REDIS_REST_TOKEN")
        self.engine_type = "SQLITE_HYDRATED"

        # Attempt GCP Firestore Initialization
        if os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT"):
            try:
                from google.cloud import firestore
                project = os.getenv("GCP_PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
                self.firestore_db = firestore.Client(project=project)
                self.engine_type = "GCP_FIRESTORE"
                logger.info(f"Initialized Distributed Checksum Cache on GCP Firestore (Project: {project})")
            except Exception as e:
                logger.info(f"Firestore notice ({str(e)}). Defaulting to Persistent SQLite Cache engine.")

        # Attempt Redis REST Initialization if configured
        if self.redis_url and self.redis_token and self.engine_type != "GCP_FIRESTORE":
            self.engine_type = "REDIS_REST"
            logger.info("Initialized Distributed Checksum Cache on Redis REST Engine.")

        # Initialize SQLite fallback database
        self._init_sqlite()

    def _init_sqlite(self):
        """Creates SQLite table for local persistent hash storage."""
        try:
            os.makedirs(self.db_dir, exist_ok=True)
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS checksums (
                        hash TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
            self.hydrate_local_memory()
        except Exception as e:
            logger.warning(f"SQLite cache initialization notice ({str(e)})")

    def hydrate_local_memory(self):
        """Auto-hydrates in-memory set from SQLite persistent store on cold start."""
        try:
            if os.path.exists(self.sqlite_path):
                with sqlite3.connect(self.sqlite_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT hash FROM checksums")
                    rows = cursor.fetchall()
                    for r in rows:
                        self.local_memory_set.add(r[0])
            logger.info(f"Hydrated Checksum Cache memory with {len(self.local_memory_set)} persistent hashes.")
        except Exception as e:
            logger.warning(f"Cache memory hydration notice ({str(e)})")

    def is_duplicate(self, text: str, custom_key: Optional[str] = None) -> bool:
        """
        Fast O(1) multi-layer check to determine if document text/key has already been indexed.
        Checks: In-Memory -> SQLite -> GCP Firestore -> Redis REST API.
        """
        raw_key = (custom_key or text).strip()
        chunk_hash = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]

        # Layer 1: In-Memory Set ($O(1)$ sub-millisecond check)
        if chunk_hash in self.local_memory_set:
            return True

        # Layer 2: GCP Firestore Check
        if self.firestore_db is not None:
            try:
                doc_ref = self.firestore_db.collection("rag_checksums").document(chunk_hash)
                if doc_ref.get().exists:
                    self.local_memory_set.add(chunk_hash)
                    return True
            except Exception as e:
                logger.warning(f"Firestore duplicate lookup notice ({str(e)})")

        # Layer 3: Redis REST Check
        if self.engine_type == "REDIS_REST":
            try:
                req = urllib.request.Request(
                    f"{self.redis_url}/EXISTS/{chunk_hash}",
                    headers={"Authorization": f"Bearer {self.redis_token}"}
                )
                with urllib.request.urlopen(req) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                    if data.get("result") == 1:
                        self.local_memory_set.add(chunk_hash)
                        return True
            except Exception as e:
                logger.warning(f"Redis REST lookup notice ({str(e)})")

        return False

    def add(self, text: str, custom_key: Optional[str] = None) -> str:
        """
        Registers new document payload hash across in-memory set, SQLite, and GCP Firestore / Redis.
        """
        raw_key = (custom_key or text).strip()
        chunk_hash = hashlib.sha256(raw_key.encode('utf-8')).hexdigest()[:16]

        # 1. Update In-Memory Set
        self.local_memory_set.add(chunk_hash)

        # 2. Update SQLite Persistent Storage
        try:
            with sqlite3.connect(self.sqlite_path) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT OR IGNORE INTO checksums (hash) VALUES (?)", (chunk_hash,))
                conn.commit()
        except Exception as e:
            logger.warning(f"SQLite cache write notice ({str(e)})")

        # 3. Update GCP Firestore
        if self.firestore_db is not None:
            try:
                doc_ref = self.firestore_db.collection("rag_checksums").document(chunk_hash)
                doc_ref.set({
                    "hash": chunk_hash,
                    "created_at": "AUTO"
                })
            except Exception as e:
                logger.warning(f"Firestore cache write notice ({str(e)})")

        # 4. Update Redis REST
        if self.engine_type == "REDIS_REST":
            try:
                req = urllib.request.Request(
                    f"{self.redis_url}/SET/{chunk_hash}/1",
                    headers={"Authorization": f"Bearer {self.redis_token}"}
                )
                urllib.request.urlopen(req)
            except Exception as e:
                logger.warning(f"Redis REST cache write notice ({str(e)})")

        return chunk_hash


# Singleton Instance
checksum_cache = ChecksumCacheEngine()
