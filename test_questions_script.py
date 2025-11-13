#!/usr/bin/env python3
"""
Test Script: Test all questions in CSV file using RAG system
Compare model output with reference answers
"""

import os
import sys
import time
from typing import Any, Dict, List

import pandas as pd

# Add project path to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

try:
    from rag.chain import RAGChain
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure you are running this script from the project root directory")
    sys.exit(1)


def load_test_questions(csv_file_path: str) -> List[Dict[str, Any]]:
    """Load test questions from CSV file"""
    questions = []
    try:
        df = pd.read_csv(csv_file_path)
        for _, row in df.iterrows():
            questions.append(
                {
                    "question_id": row["Question_ID"],
                    "category": row["Category"],
                    "question": row["Question"],
                    "expected_answer": row["Expected_Answer"],
                    "evaluation_criteria": row["Evaluation_Criteria"],
                }
            )
        print(f"Successfully loaded {len(questions)} test questions")
        return questions
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        print(f"Error loading CSV file: {e}")
        return []


def evaluate_answer_correctness(model_answer: str, expected_answer: str) -> str:
    """Evaluate if model answer is correct"""
    # Simple string matching evaluation
    model_answer_lower = model_answer.lower()
    expected_answer_lower = expected_answer.lower()

    # Check if key information matches
    key_phrases = [
        "85°c",
        "85°c",
        "85c",
        "85 c",  # Temperature
        "2 gb",
        "2gb",
        "2 gb",  # Memory
        "5 tops",
        "5tops",
        "5 tops",  # NPU performance
        "1.7a",
        "1.7 a",
        "1.7a",  # Power consumption
        "32 gb",
        "32gb",
        "32 gb",  # Storage
        "105°c",
        "105°c",
        "105c",
        "105 c",  # Temperature
        "me-driver-ctl --enable-npu --mode=performance",  # NPU command
    ]

    # Check if contains key information
    for phrase in key_phrases:
        if phrase in expected_answer_lower and phrase in model_answer_lower:
            return "correct"

    # If key information doesn't match, return needs manual review
    return "needs_review"


def test_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Test all questions and return results"""
    results = []

    # Create RAG chain
    print("Initializing RAG system...")
    try:
        rag_chain = RAGChain()
        print("RAG system initialized successfully")
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"RAG system initialization failed: {e}")
        return results

    for i, question_data in enumerate(questions, 1):
        print(f"\nTest question {i}/{len(questions)}: {question_data['question_id']} - {question_data['category']}")
        print(f"Question: {question_data['question']}")

        try:
            # Use RAG system to answer question
            start_time = time.time()
            response = rag_chain.query(question_data["question"])
            end_time = time.time()

            response_time = end_time - start_time

            # Extract answer from response
            model_answer = response.get("answer", "")

            # Evaluate answer correctness
            correctness = evaluate_answer_correctness(
                model_answer, question_data["expected_answer"]
            )

            result = {
                "question_id": question_data["question_id"],
                "category": question_data["category"],
                "question": question_data["question"],
                "expected_answer": question_data["expected_answer"],
                "model_answer": str(model_answer),
                "response_time": response_time,
                "evaluation_criteria": question_data["evaluation_criteria"],
                "status": "success",
                "correctness": correctness,
            }

            print(f"Answer successful (time: {response_time:.2f}s, correctness: {correctness})")

        except (ValueError, RuntimeError, TimeoutError) as e:
            print(f"Answer failed: {e}")
            result = {
                "question_id": question_data["question_id"],
                "category": question_data["category"],
                "question": question_data["question"],
                "expected_answer": question_data["expected_answer"],
                "model_answer": f"Error: {str(e)}",
                "response_time": 0,
                "evaluation_criteria": question_data["evaluation_criteria"],
                "status": "error",
                "correctness": "error",
            }

        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]], output_file: str = "test_results.csv"):
    """Save test results to CSV file"""
    try:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"\nTest results saved to: {output_file}")
    except (ValueError, FileNotFoundError, PermissionError) as e:
        print(f"Error saving results: {e}")


def print_comparison(results: List[Dict[str, Any]]):
    """Print comparison results"""
    print("\n" + "=" * 80)
    print("Test Results Comparison")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")

    print(f"Total questions: {len(results)}")
    print(f"Successfully answered: {success_count}")
    print(f"Failed answers: {error_count}")

    for result in results:
        print(f"\n{'='*60}")
        print(f"Question ID: {result['question_id']}")
        print(f"Category: {result['category']}")
        print(f"Status: {result['status']}")
        if result["status"] == "success":
            print(f"Response time: {result['response_time']:.2f}s")

        print(f"\nQuestion: {result['question']}")
        print(f"\nExpected answer:")
        print(f"{result['expected_answer']}")
        print(f"\nModel answer:")
        print(f"{result['model_answer']}")
        print(f"\nEvaluation criteria: {result['evaluation_criteria']}")
        print(f"{'='*60}")


def main():
    """Main function"""
    # CSV file path
    csv_file_path = "test-questions.csv"

    # Check if file exists
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found: {csv_file_path}")
        print("Please ensure the file path is correct")
        return

    # Load test questions
    print("Loading test questions...")
    questions = load_test_questions(csv_file_path)

    if not questions:
        print("No test questions found, exiting program")
        return

    # Test questions
    print("\nStarting tests...")
    results = test_questions(questions)

    # Save results
    save_results(results)

    # Print comparison
    print_comparison(results)

    # Generate summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)

    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    correct_count = sum(1 for r in results if r.get("correctness") == "correct")
    needs_review_count = sum(1 for r in results if r.get("correctness") == "needs_review")

    total_time = sum(r["response_time"] for r in results if r["status"] == "success")
    avg_time = total_time / success_count if success_count > 0 else 0

    print(f"Total questions: {len(results)}")
    print(f"Successfully answered: {success_count}")
    print(f"Failed answers: {error_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Needs manual review: {needs_review_count}")
    print(f"Average response time: {avg_time:.2f}s")
    print(f"Total test time: {total_time:.2f}s")

    # Accuracy rate statistics
    if success_count > 0:
        accuracy_rate = (correct_count / success_count) * 100
        print(f"Auto-evaluation accuracy rate: {accuracy_rate:.1f}%")


if __name__ == "__main__":
    main()
