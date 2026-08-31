"""
AWS Bedrock Intelligent Model Router
Author: Raja Chakraborty

Dynamically routes incoming prompts & retrieved RAG contexts to the optimal AWS Bedrock model
(Claude 3.5 Sonnet vs. Llama 3 vs. Titan) based on query complexity, latency targets, and token budget.
Includes Mock Bedrock Mode for open-source execution without requiring active AWS keys.
"""

import json
import logging
from typing import Dict, Any, Optional
from config import (
    get_bedrock_client,
    BEDROCK_PRIMARY_MODEL_ID,
    BEDROCK_FAST_MODEL_ID,
)

logger = logging.getLogger("rag-lakehouse-bedrock")


def determine_query_complexity(prompt: str) -> str:
    """
    Analyzes prompt heuristics to categorize query complexity:
    - 'COMPLEX': Code generation, architectural design, deep reasoning -> Route to Claude 3.5 Sonnet
    - 'STANDARD': Summarization, Q&A, retrieval lookup -> Route to Llama 3 / Titan
    """
    complex_keywords = ["architect", "design", "code", "algorithm", "compare", "refactor", "security", "why", "how does"]
    prompt_lower = prompt.lower()

    if len(prompt.split()) > 30 or any(keyword in prompt_lower for keyword in complex_keywords):
        return "COMPLEX"
    return "STANDARD"


def route_prompt_to_bedrock(
    prompt: str,
    context: Optional[str] = None,
    override_model_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Routes query to AWS Bedrock runtime. Selects model automatically based on complexity.
    If no AWS client is available, executes Mock Bedrock Mode to ensure open GitHub repo safety.
    """
    bedrock = get_bedrock_client()
    complexity = determine_query_complexity(prompt)

    selected_model_id = override_model_id or (
        BEDROCK_PRIMARY_MODEL_ID if complexity == "COMPLEX" else BEDROCK_FAST_MODEL_ID
    )

    full_prompt = f"Context:\n{context}\n\nQuery:\n{prompt}" if context else prompt

    logger.info(f"Routing Prompt [Complexity: {complexity}] -> Model: [{selected_model_id}]")

    # If active AWS Bedrock client exists, invoke AWS Bedrock Runtime
    if bedrock is not None:
        try:
            if "anthropic" in selected_model_id:
                body = json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 1024,
                    "messages": [{"role": "user", "content": full_prompt}],
                })
            else:
                body = json.dumps({
                    "prompt": full_prompt,
                    "max_gen_len": 512,
                    "temperature": 0.5,
                })

            response = bedrock.invoke_model(
                modelId=selected_model_id,
                body=body,
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response["body"].read())

            if "anthropic" in selected_model_id:
                text_out = response_body["content"][0]["text"]
            else:
                text_out = response_body.get("generation", str(response_body))

            return {
                "status": "success",
                "model_used": selected_model_id,
                "complexity": complexity,
                "response_text": text_out,
                "mode": "AWS_BEDROCK_LIVE",
            }
        except Exception as e:
            logger.error(f"AWS Bedrock API call failed: {str(e)}. Falling back to Mock Mode.")

    # Mock Mode Fallback (Safe for public GitHub repos without AWS API keys)
    mock_response = (
        f"[Mock Bedrock Engine — {selected_model_id}]\n"
        f"Retrieved Context: {context[:150]}...\n\n"
        f"Answer: Based on the knowledge base, '{prompt}' is handled by our distributed real-time pipeline."
    ) if context else (
        f"[Mock Bedrock Engine — {selected_model_id}]\n"
        f"Answer to '{prompt}': Processed with {complexity} complexity routing."
    )

    return {
        "status": "success",
        "model_used": selected_model_id,
        "complexity": complexity,
        "response_text": mock_response,
        "mode": "MOCK_BEDROCK_SAFE",
    }


if __name__ == "__main__":
    res1 = route_prompt_to_bedrock("Explain the architecture of RoaringBitmap seat maps", context="RoaringBitmap is a bit set representation.")
    print("Test 1 Result:", res1)
