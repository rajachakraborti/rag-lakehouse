"""
Configuration Manager for RAG-Lakehouse Engine (GCP & Open Source)
Author: Raja Chakraborty

Handles environment variables, GCP Vertex AI / Gemini credential verification,
PyTorch compute device selection, and safe mock fallback mode for open-source execution.
"""

import os
import logging
import torch
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rag-lakehouse")

# Load environment variables from .env file if present
load_dotenv()

# Security & GCP Configuration
GCP_PROJECT = os.getenv("GCP_PROJECT", "my-gcp-project")
GCP_REGION = os.getenv("GCP_REGION", "us-central1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GCP_PRIMARY_MODEL_ID = os.getenv("GCP_PRIMARY_MODEL_ID", "gemini-1.5-pro")
GCP_FAST_MODEL_ID = os.getenv("GCP_FAST_MODEL_ID", "gemini-1.5-flash")

# Local Storage & Computation Settings
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", "./chroma_storage")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "RAG-Lakehouse-Pipeline")

# PyTorch Device Selection
def get_pytorch_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"PyTorch Compute Device Selected: [{device.upper()}]")
    return device

TORCH_DEVICE = get_pytorch_device()

# GCP Gemini Client Initializer with Mock Fallback for Open Repositories
def get_gcp_gemini_client():
    """
    Initializes Google Generative AI (Gemini) client if GEMINI_API_KEY exists.
    If no API key is configured, returns None to trigger Mock GCP Mode
    (allowing safe public GitHub execution without GCP account dependencies).
    """
    if GEMINI_API_KEY and "your_gemini" not in GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=GEMINI_API_KEY)
            logger.info("✅ GCP Gemini Client initialized successfully with environment credentials.")
            return genai
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize GCP Gemini client ({str(e)}). Defaulting to Mock Mode.")
            return None
    else:
        logger.info("ℹ️ No GCP API key detected in environment. Running in [Mock GCP Mode] (Safe for Open GitHub).")
        return None
