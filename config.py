"""
Configuration & Security Manager for RAG-Lakehouse Engine
Author: Raja Chakraborty

Handles environment variables, AWS Bedrock credential verification,
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

# Security & AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

BEDROCK_PRIMARY_MODEL_ID = os.getenv("BEDROCK_PRIMARY_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
BEDROCK_FAST_MODEL_ID = os.getenv("BEDROCK_FAST_MODEL_ID", "meta.llama3-8b-instruct-v1:0")
BEDROCK_EMBEDDING_MODEL_ID = os.getenv("BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")

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

# AWS Bedrock Client Initializer with Mock Fallback for Open Repositories
def get_bedrock_client():
    """
    Initializes boto3 Bedrock client if valid AWS credentials exist.
    If no AWS credentials are configured, returns None to trigger Mock Bedrock Mode
    (allowing safe public GitHub execution without AWS account dependencies).
    """
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and "your_aws" not in AWS_ACCESS_KEY_ID:
        try:
            import boto3
            client = boto3.client(
                "bedrock-runtime",
                region_name=AWS_REGION,
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            )
            logger.info("✅ AWS Bedrock Client initialized successfully with environment credentials.")
            return client
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize AWS Bedrock client ({str(e)}). Defaulting to Mock Mode.")
            return None
    else:
        logger.info("ℹ️ No AWS credentials detected in environment. Running in [Mock Bedrock Mode] (Safe for Open GitHub).")
        return None
