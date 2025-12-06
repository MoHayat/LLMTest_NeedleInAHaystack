#!/usr/bin/env python3
"""
Simple script to test the Groq provider directly (without the full NIAH benchmark).

This verifies that the Groq adapter is working correctly by making a simple API call.
"""

import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

# Verify required environment variables
if not os.getenv("NIAH_MODEL_API_KEY"):
    print("ERROR: NIAH_MODEL_API_KEY environment variable is not set.")
    print("Please set it with: export NIAH_MODEL_API_KEY='your-groq-api-key'")
    sys.exit(1)

from needlehaystack.providers import Groq


async def test_groq_provider():
    """Test the Groq provider directly."""

    MODEL_NAME = "openai/gpt-oss-20b"

    print("=" * 60)
    print("Groq Provider Direct Test")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    # Initialize the Groq provider
    print("\nInitializing Groq provider...")
    provider = Groq(
        model_name=MODEL_NAME,
        model_kwargs={
            "max_tokens": 300,
            "temperature": 0.0,
        },
    )

    # Test tokenization
    print("\n--- Testing Tokenization ---")
    test_text = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    tokens = provider.encode_text_to_tokens(test_text)
    print(f"Original text: {test_text}")
    print(f"Token count: {len(tokens)}")
    decoded = provider.decode_tokens(tokens)
    print(f"Decoded text: {decoded}")

    # Test prompt generation
    print("\n--- Testing Prompt Generation ---")
    context = """Paul Graham wrote about startups and technology.
    
The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.

He also wrote about programming languages and venture capital."""

    question = "What is the best thing to do in San Francisco?"
    prompt = provider.generate_prompt(context, question)
    print(f"Generated prompt structure:")
    for msg in prompt:
        print(f"  [{msg['role']}]: {msg['content'][:100]}...")

    # Test model evaluation
    print("\n--- Testing Model Response ---")
    print("Sending request to Groq...")
    try:
        response = await provider.evaluate_model(prompt)
        print(f"\nGroq's Response:")
        print("-" * 40)
        print(response)
        print("-" * 40)

        # Check if the response contains the needle
        if "sandwich" in response.lower() or "dolores park" in response.lower():
            print("\n✅ SUCCESS: The model found the needle in the haystack!")
        else:
            print(
                "\n⚠️ The model's response doesn't seem to contain the expected answer."
            )

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_groq_provider())
