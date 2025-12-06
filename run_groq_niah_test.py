#!/usr/bin/env python3
"""
Script to run a Needle in a Haystack (NIAH) test using the Groq provider.

This script runs a simple test to verify the Groq adapter is working correctly.
It tests a single context length and depth percentage for quick validation.

Usage:
    python run_groq_niah_test.py

Environment Variables Required:
    - NIAH_MODEL_API_KEY: Your Groq API key (used for both model and evaluator)

Results will be saved to the 'results' directory.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path so we can import the needlehaystack module
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify required environment variables
if not os.getenv("NIAH_MODEL_API_KEY"):
    print("ERROR: NIAH_MODEL_API_KEY environment variable is not set.")
    print("Please set it with: export NIAH_MODEL_API_KEY='your-groq-api-key'")
    sys.exit(1)

from needlehaystack import LLMNeedleHaystackTester
from needlehaystack.providers import Groq
from needlehaystack.evaluators import GroqEvaluator


def main():
    """Run a simple NIAH test with the Groq provider."""

    # Configuration
    MODEL_NAME = "openai/gpt-oss-20b"  # or "openai/gpt-oss-20b", "mixtral-8x7b-32768"

    # Test needle and question
    NEEDLE = "\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n"
    RETRIEVAL_QUESTION = "What is the best thing to do in San Francisco?"

    # Test parameters - using small values for quick testing
    CONTEXT_LENGTHS = [
        1000,
        10000,
        19000,
        28000,
        37000,
        46000,
        55000,
        64000,
        73000,
        82000,
        91000,
        100000,
        109000,
        118000,
        128000,
    ]  # Token lengths to test
    DOCUMENT_DEPTH_PERCENTS = [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]  # Where to place the needle (50 = middle)

    print("=" * 60)
    print("Needle In A Haystack Test - Groq Provider")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Context Length(s): {CONTEXT_LENGTHS}")
    print(f"Document Depth(s): {DOCUMENT_DEPTH_PERCENTS}%")
    print(f"Needle: {NEEDLE.strip()}")
    print(f"Question: {RETRIEVAL_QUESTION}")
    print("=" * 60)

    # Initialize the Groq provider
    print("\nInitializing Groq provider...")
    model_to_test = Groq(
        model_name=MODEL_NAME,
        model_kwargs={
            "max_tokens": 300,
            "temperature": 0.0,  # Use 0 for deterministic outputs
        },
    )

    # Initialize the evaluator (uses Groq to score the response - same API key)
    print("Initializing Groq evaluator...")
    evaluator = GroqEvaluator(
        model_name="llama-3.3-70b-versatile",  # Use a capable model for evaluation
        question_asked=RETRIEVAL_QUESTION,
        true_answer=NEEDLE,
    )

    # Create the tester
    print("Creating NIAH tester...")
    tester = LLMNeedleHaystackTester(
        model_to_test=model_to_test,
        evaluator=evaluator,
        needle=NEEDLE,
        haystack_dir="PaulGrahamEssays",
        retrieval_question=RETRIEVAL_QUESTION,
        context_lengths=CONTEXT_LENGTHS,
        document_depth_percents=DOCUMENT_DEPTH_PERCENTS,
        results_version=1,
        save_results=True,
        save_contexts=False,  # Set to True to save the full contexts
        print_ongoing_status=True,
        num_concurrent_requests=1,  # Groq has rate limits, so keep this low
    )

    # Run the test
    print("\nStarting NIAH test...")
    print("-" * 60)
    tester.start_test()

    print("\n" + "=" * 60)
    print("Test complete! Check the 'results' directory for output.")
    print("=" * 60)


if __name__ == "__main__":
    main()
