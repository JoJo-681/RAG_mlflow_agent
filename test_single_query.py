#!/usr/bin/env python3
"""
Test script for single query testing of the RAG system.

This script allows you to test the RAG system with a single query
and get detailed response information.
"""

import argparse
import json
import sys
from typing import Any, Dict

import requests


class RAGQueryTester:
    """Test class for single query testing of RAG system."""

    def __init__(self, mlflow_url: str = "http://localhost:5001"):
        """
        Initialize the RAG query tester.

        Args:
            mlflow_url: URL of the MLflow serving endpoint
        """
        self.mlflow_url = mlflow_url.rstrip("/")
        self.invocations_url = f"{self.mlflow_url}/invocations"

    def test_single_query(self, query: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Test a single query against the RAG system.

        Args:
            query: The question to ask
            verbose: Whether to print detailed output

        Returns:
            Dictionary containing the response data
        """
        # Prepare the request payload
        payload = {"dataframe_split": {"columns": ["query"], "data": [[query]]}}

        if verbose:
            print(f"[TEST] Testing query: '{query}'")
            print(f"[SEND] Sending request to: {self.invocations_url}")

        try:
            # Send the request
            response = requests.post(
                self.invocations_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=30,
            )

            # Check if request was successful
            if response.status_code == 200:
                result = response.json()

                if verbose:
                    self._print_detailed_result(query, result)

                return {
                    "success": True,
                    "query": query,
                    "response": result,
                    "status_code": response.status_code,
                }
            else:
                error_msg = f"Request failed with status {response.status_code}: {response.text}"
                if verbose:
                    print(f"[ERROR] {error_msg}")

                return {
                    "success": False,
                    "query": query,
                    "error": error_msg,
                    "status_code": response.status_code,
                }

        except requests.exceptions.ConnectionError:
            error_msg = f"[ERROR] Cannot connect to MLflow server at {self.mlflow_url}"
            if verbose:
                print(f"{error_msg} - Cannot connect to MLflow server: {self.mlflow_url}")
                print("[INFO] Make sure the MLflow server is running with: python deploy_rag_with_mlflow.py")
            return {"success": False, "query": query, "error": error_msg, "status_code": None}
        except requests.exceptions.Timeout:
            error_msg = "[ERROR] Request timed out"
            if verbose:
                print(f"{error_msg} - Request timeout")
            return {"success": False, "query": query, "error": error_msg, "status_code": None}
        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {str(e)}"
            if verbose:
                print(f"{error_msg} - Unexpected error: {str(e)}")
            return {"success": False, "query": query, "error": error_msg, "status_code": None}

    def _print_detailed_result(self, query: str, result: Dict[str, Any]) -> None:
        """Print detailed result information."""
        print("\n" + "=" * 60)
        print("[DETAILS] QUERY RESULT DETAILS")
        print("=" * 60)
        print(f"[QUERY] Query: {query}")
        print("-" * 60)

        # Extract the main response - use get() method to avoid KeyError
        predictions = result.get("predictions", {})

        if predictions:
            # Handle different response formats
            if isinstance(predictions, dict):
                # Single prediction dictionary format
                print(f"[ANSWER] Answer: {predictions.get('answer', 'No answer found')}")
                print(f"[STATUS] Status: {predictions.get('status', 'N/A')}")
                print(f"[DOCS] Source Documents: {predictions.get('source_documents_count', 'N/A')}")

                # Print source documents if available
                source_docs = predictions.get("source_documents", [])
                if source_docs:
                    print(f"[RETRIEVED] Retrieved Documents: {len(source_docs)}")
                    for i, doc in enumerate(source_docs, 1):
                        print(f"   {i}. {doc.get('metadata', {}).get('source', 'Unknown')}")

            elif isinstance(predictions, list) and len(predictions) > 0:
                # List of predictions format
                prediction = predictions[0]
                if isinstance(prediction, dict):
                    print(f"[ANSWER] Answer: {prediction.get('answer', 'No answer found')}")
                    print(f"[STATUS] Status: {prediction.get('status', 'N/A')}")
                    print(f"[DOCS] Source Documents: {prediction.get('source_documents_count', 'N/A')}")
                else:
                    print(f"[RESPONSE] Response: {prediction}")
            else:
                print(f"[RESPONSE] Response: {predictions}")
        else:
            print("[ERROR] No predictions found in response")
            print(f"[RESPONSE] Full response: {result}")

        print("=" * 60)
        print("[SUCCESS] Query completed successfully!\n")

    def interactive_mode(self) -> None:
        """Run in interactive mode for multiple queries."""
        print("[MODE] RAG Query Tester - Interactive Mode")
        print("[INFO] Type 'quit' or 'exit' to stop")
        print("-" * 40)

        while True:
            try:
                query = input("\n[INPUT] Enter your question: ").strip()

                if query.lower() in ["quit", "exit", "q"]:
                    print("[EXIT] Goodbye!")
                    break

                if not query:
                    print("[WARNING] Please enter a question")
                    continue

                self.test_single_query(query, verbose=True)

            except KeyboardInterrupt:
                print("\n[EXIT] Goodbye!")
                break
            except Exception as e:
                print(f"[ERROR] Error: {e}")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description="Test single queries against RAG system")
    parser.add_argument(
        "query", nargs="?", help="The question to ask (if not provided, runs in interactive mode)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:5001",
        help="MLflow serving URL (default: http://localhost:5001)",
    )

    args = parser.parse_args()

    tester = RAGQueryTester(args.url)

    if args.query:
        # Single query mode
        result = tester.test_single_query(args.query, verbose=True)

        if result["success"]:
            print("[SUCCESS] Test completed successfully!")
        else:
            print("[FAILURE] Test failed!")
            sys.exit(1)
    else:
        # Interactive mode
        tester.interactive_mode()


if __name__ == "__main__":
    main()
