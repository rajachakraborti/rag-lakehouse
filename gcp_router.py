"""
GCP Vertex AI & Gemini Intelligent Model Router
Author: Raja Chakraborty

Dynamically routes incoming prompts & retrieved RAG contexts to the optimal GCP model
(Gemini 1.5 Pro vs. Gemini 1.5 Flash) based on query complexity, latency targets, and compute budget.
Includes Mock GCP Mode for open-source execution without requiring active API keys.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from config import (
    get_gcp_gemini_client,
    GCP_PRIMARY_MODEL_ID,
    GCP_FAST_MODEL_ID,
)

logger = logging.getLogger("rag-lakehouse-gcp")


def determine_query_complexity(prompt: str) -> str:
    """
    Analyzes prompt heuristics to categorize query complexity:
    - 'COMPLEX': Code generation, architectural design, deep reasoning -> Route to Gemini 1.5 Pro
    - 'STANDARD': Summarization, Q&A, retrieval lookup -> Route to Gemini 1.5 Flash
    """
    complex_keywords = ["architect", "design", "code", "algorithm", "compare", "refactor", "security", "why", "how does"]
    prompt_lower = prompt.lower()

    if len(prompt.split()) > 30 or any(keyword in prompt_lower for keyword in complex_keywords):
        return "COMPLEX"
    return "STANDARD"


def route_prompt_to_gcp(
    prompt: str,
    context: Optional[str] = None,
    override_model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Routes query to GCP Vertex AI runtime. Selects model automatically based on complexity.
    If no GCP client is available, executes Mock GCP Mode to ensure open GitHub repo safety.
    """
    client = get_gcp_gemini_client()
    complexity = determine_query_complexity(prompt)

    selected_model_id = override_model_id or (
        GCP_PRIMARY_MODEL_ID if complexity == "COMPLEX" else GCP_FAST_MODEL_ID
    )

    full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}" if context else prompt

    logger.info(f"Routing Prompt [Complexity: {complexity}] -> GCP Model: [{selected_model_id}]")

    # If active GCP Gemini client exists, invoke Vertex AI / Gemini API
    if client is not None:
        try:
            model = client.GenerativeModel(selected_model_id)
            response = model.generate_content(full_prompt)
            return {
                "status": "success",
                "model_used": selected_model_id,
                "complexity": complexity,
                "response_text": response.text,
                "mode": "GCP_VERTEX_AI_LIVE",
            }
        except Exception as e:
            logger.error(f"GCP Vertex AI API call failed: {str(e)}. Falling back to Mock Mode.")

    # Dynamic Context Synthesis Engine (Safe for open GitHub repos without GCP API keys)
    if context and len(context.strip()) > 0:
        clean_ctx = context.strip()
        mock_response = (
            f"[GCP Gemini Engine — {selected_model_id}]\n"
            f"Based on retrieved lakehouse knowledge base context:\n\n"
            f"\"{clean_ctx}\"\n\n"
            f"Synthesized Answer: According to the lakehouse documents, {clean_ctx}"
        )
    else:
        mock_response = (
            f"[GCP Gemini Engine — {selected_model_id}]\n"
            f"No specific knowledge base context was found for '{prompt}'. Active topics in the lakehouse include: RoaringBitmap seat maps, PromptShield PII redaction, Cloud Run scaling, and Raja Chakraborty's engineering profile."
        )

    return {
        "status": "success",
        "model_used": selected_model_id,
        "complexity": complexity,
        "response_text": mock_response,
        "mode": "MOCK_GCP_SAFE",
    }


if __name__ == "__main__":
    res1 = route_prompt_to_gcp("Explain the architecture of RoaringBitmap seat maps", context="RoaringBitmap is a bit set representation.")
    print("GCP Router Result:", res1)
