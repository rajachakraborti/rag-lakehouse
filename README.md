# 🚀 RAG-Lakehouse — AI Engine, Multi-Cloud Router & Data Pipeline

> **An enterprise-grade, multi-cloud AI platform** built with **PySpark, PyTorch, Vector DBs, AWS Bedrock, GCP Cloud Run, Pulumi IaC (Python), and Model Context Protocol (MCP)**.
> Demonstrates end-to-end data lakehouse processing, hybrid vector search, dynamic model routing, and agentic tool integration.

---

## 🔒 Security & Open GitHub Safeguards

This repository is designed to be **100% open on GitHub with zero secret leak risk**:
1. **Zero Hardcoded Secrets:** All credentials, model IDs, and database paths are loaded dynamically from environment variables (`.env`). `.env` is explicitly ignored in `.gitignore`.
2. **`.env.example` Template:** Provided as a dummy configuration for quick setup.
3. **Safe Mock Fallback Mode:** If no AWS/GCP credentials exist in the environment, the engine automatically enters **Mock Bedrock/GCP Mode** — allowing integration tests to execute completely without requiring active cloud subscriptions or keys.

---

## 📐 Platform Architecture

```mermaid
graph TD
    Docs["📄 Unstructured Docs / Logs / Data"] --> Spark["⚡ PySpark Pipeline\n(Distributed Chunking & DataFrame Ops)"]
    Spark --> PyTorch["🔥 PyTorch & Transformers\n(Vector Embedding Engine)"]
    PyTorch --> VectorDB["🗄️ Vector DB (ChromaDB / Qdrant)\n(Metadata Filtering & Similarity Index)"]

    Agent["🤖 AI Agent (Claude Desktop / Cursor / Custom Agent)"] -->|MCP Protocol stdio/SSE| MCPServer["🔌 MCP Tool Server (mcp_server.py)\n(Exposes RAG & Model Tools)"]
    MCPServer --> Engine["🧠 Hybrid RAG Engine (rag_engine.py)"]
    Engine <-->|Vector Retrieval| VectorDB
    Engine --> Router["🔀 Multi-Cloud LLM Router (bedrock_router.py)\n(Complexity Heuristics)"]

    Router -->|Complex Queries| Sonnet["☁️ AWS Bedrock: Claude 3.5 Sonnet"]
    Router -->|Fast/Low Cost| Gemini["☁️ GCP Vertex AI: Gemini 1.5 Flash / Llama 3"]
```

---

## 🏛️ Infrastructure as Code (Pulumi in Python)

Infrastructure is defined in pure Python using **Pulumi** located in `pulumi/`:
- **Google Cloud Run v2 (`cloud_run_service`)**: Auto-scaling serverless backend (scales to `min_instances = 0` for **$0.00 idle cost**).
- **Google Cloud Storage (`lakehouse_bucket`)**: Secure document data lake storage.
- **IAM Least Privilege**: Custom service account and granular role bindings.

---

## 🛠️ Technology Stack & Role Breakdown

| Technology | Architectural Role |
|---|---|
| **PySpark** | **Distributed Ingestion:** Parallelizes text extraction, chunking, and metadata mapping across Spark DataFrames for large document volumes. |
| **PyTorch** | **Compute Engine:** Generates high-dimensional vector embeddings (`sentence-transformers`) utilizing available GPU/MPS/CPU compute hardware. |
| **Vector DB (ChromaDB)** | **Persistent Storage:** Indexes vector embeddings with metadata payload filtering for sub-50ms hybrid retrieval. |
| **Hybrid RAG Engine** | **Context Retrieval:** Combines vector similarity scoring with metadata filtering and formats context for LLM synthesis. |
| **AWS Bedrock / GCP Router** | **Intelligent Model Selection:** Evaluates query complexity and dynamically routes prompts between Claude 3.5 Sonnet (complex reasoning) and Gemini 1.5 / Llama 3 (fast Q&A). |
| **Model Context Protocol (MCP)** | **Agentic Tool Server:** Exposes RAG search, document ingestion, and Bedrock routing as standard MCP tools for AI agents. |
| **Pulumi (Python)** | **Infrastructure as Code:** Provisions serverless Cloud Run v2, GCS buckets, and IAM roles in native Python code. |

---

## 🚀 Quickstart & Setup Guide

### 1. Environment Setup
Clone the repository and install dependencies:

```bash
git clone https://github.com/rajachakraborti/rag-lakehouse.git
cd rag-lakehouse
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

### 3. Run Automated Integration Tests
Run the test suite verifying Spark chunking, PyTorch indexing, model routing, and MCP tools:

```bash
python test_pipeline.py
```

### 4. Deploy Infrastructure via Pulumi (Python)
```bash
cd pulumi
pip install -r requirements.txt
pulumi up
```

---

## 🗣️ Senior / Lead AI Engineering Interview Talking Points

- **Python-Native IaC with Pulumi:** *"Instead of maintaining domain-specific HCL files, I declared our GCP Cloud Run, GCS data lake, and IAM security resources using Pulumi in Python (`pulumi/__main__.py`), making infrastructure fully unit-testable and aligned with our software engineering pipeline."*
- **Big Data + AI Bridge:** *"I built an ingestion pipeline using PySpark for parallel text processing and chunking, bridging distributed data engineering with PyTorch vector embeddings and ChromaDB indexing."*
- **Cost & Serverless Efficiency:** *"Configured GCP Cloud Run with `min-instances=0` so the API backend scales down to zero when idle, consuming zero compute budget while maintaining instant elasticity under load."*
