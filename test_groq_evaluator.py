#!/usr/bin/env python3
"""
Test the Groq evaluator directly to verify it works.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("NIAH_MODEL_API_KEY"):
    print("ERROR: NIAH_MODEL_API_KEY environment variable is not set.")
    sys.exit(1)

from needlehaystack.evaluators import GroqEvaluator


def test_groq_evaluator():
    """Test the Groq evaluator directly."""

    print("=" * 60)
    print("Testing Groq Evaluator")
    print("=" * 60)

    NEEDLE = "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    QUESTION = "What is the best thing to do in San Francisco?"

    # Initialize the evaluator
    print("\nInitializing Groq evaluator...")
    evaluator = GroqEvaluator(
        model_name="llama-3.3-70b-versatile",
        question_asked=QUESTION,
        true_answer=NEEDLE,
    )

    # Test with a correct answer
    print("\n--- Test 1: Correct Answer ---")
    correct_response = "Eat a sandwich and sit in Dolores Park on a sunny day."
    print(f"Response: {correct_response}")
    try:
        score = evaluator.evaluate_response(correct_response)
        print(f"Score: {score}/10")
    except Exception as e:
        print(f"Error: {e}")

    # Test with a partially correct answer
    print("\n--- Test 2: Partial Answer ---")
    partial_response = "Eat a sandwich in San Francisco."
    print(f"Response: {partial_response}")
    try:
        score = evaluator.evaluate_response(partial_response)
        print(f"Score: {score}/10")
    except Exception as e:
        print(f"Error: {e}")

    # Test with an incorrect answer
    print("\n--- Test 3: Wrong Answer ---")
    wrong_response = "Visit the Golden Gate Bridge."
    print(f"Response: {wrong_response}")
    try:
        score = evaluator.evaluate_response(wrong_response)
        print(f"Score: {score}/10")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("Evaluator test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_groq_evaluator()
