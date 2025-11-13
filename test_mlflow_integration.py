"""Test MLflow model service using MLflow standard format with timing and full answers."""

import time

import requests


def test_mlflow_service():
    """Test MLflow service using standard MLflow format with timing."""

    print("Testing MLflow service with standard format...")

    # Test health endpoint
    print("\n1. Testing health endpoint...")
    try:
        start_time = time.time()
        response = requests.get("http://localhost:5001/ping", timeout=5)
        end_time = time.time()
        response_time = end_time - start_time  # Keep in seconds

        print(f"Status: {response.status_code}")
        print(f"Response Time: {response_time:.3f}s")
        if response.status_code == 200:
            # MLflow health endpoint returns empty response, just check status
            print("Health check passed (empty response is normal for MLflow)")
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

    # Test with MLflow standard format
    print("\n2. Testing with MLflow standard format...")
    try:
        data = {
            "dataframe_split": {
                "columns": ["query"],
                "data": [["What is the power consumption of the ECU-850b under load?"]],
            }
        }

        start_time = time.time()
        response = requests.post("http://localhost:5001/invocations", json=data, timeout=30)
        end_time = time.time()
        response_time = end_time - start_time  # Keep in seconds

        print(f"Status: {response.status_code}")
        print(f"Response Time: {response_time:.3f}s")
        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", {})
            print("Success!")
            print(f"Query: {predictions.get('query', 'N/A')}")
            print(f"Answer: {predictions.get('answer', 'N/A')}")
            print(f"Source documents: {predictions.get('source_documents_count', 'N/A')}")
            print(f"Status: {predictions.get('status', 'N/A')}")
            return True
        else:
            print(f"Error: {response.text}")
            return False

    except Exception as e:
        print(f"Request failed: {e}")
        return False


def test_multiple_queries():
    """Test multiple queries with timing and full answers."""
    print("\n3. Testing multiple queries...")

    queries = [
        "What is the maximum operating temperature for the ECU-750?",
        "How much RAM does the ECU-850 have?",
        "How do you enable the NPU on the ECU-850b?",
    ]

    for query in queries:
        print(f"\n{'='*80}")
        print(f"Query: {query}")
        print("-" * 80)
        try:
            data = {"dataframe_split": {"columns": ["query"], "data": [[query]]}}

            start_time = time.time()
            response = requests.post("http://localhost:5001/invocations", json=data, timeout=30)
            end_time = time.time()
            response_time = end_time - start_time  # Keep in seconds

            if response.status_code == 200:
                result = response.json()
                predictions = result.get("predictions", {})
                print(f"Status: {predictions.get('status', 'N/A')}")
                print(f"Response Time: {response_time:.3f}s")
                print(f"Source Documents: {predictions.get('source_documents_count', 'N/A')}")
                print(f"Full Answer:")
                print(f"   {predictions.get('answer', 'N/A')}")
            else:
                print(f"Failed: {response.text}")

        except Exception as e:
            print(f"Error: {e}")


def test_batch_queries():
    """Test batch queries with timing."""
    print("\n4. Testing batch queries...")

    try:
        data = {
            "dataframe_split": {
                "columns": ["query"],
                "data": [
                    ["What is the maximum operating temperature for the ECU-750?"],
                    ["How much RAM does the ECU-850 have?"],
                    ["How do you enable the NPU on the ECU-850b?"],
                ],
            }
        }

        start_time = time.time()
        response = requests.post("http://localhost:5001/invocations", json=data, timeout=60)
        end_time = time.time()
        response_time = end_time - start_time  # Keep in seconds

        print(f"Status: {response.status_code}")
        print(f"Batch Response Time: {response_time:.3f}s")
        print(f"Average Time per Query: {response_time/5:.3f}s")

        if response.status_code == 200:
            result = response.json()
            predictions = result.get("predictions", {})
            print("Batch query successful!")
            print(f"Batch Status: {predictions.get('status', 'N/A')}")
            print(f"Total Queries: {predictions.get('total_queries', 'N/A')}")
            print(f"Successful Queries: {predictions.get('successful_queries', 'N/A')}")
            print(f"Failed Queries: {predictions.get('failed_queries', 'N/A')}")

            # Display individual results
            batch_results = predictions.get("batch_results", [])
            if batch_results:
                print("\nIndividual Results:")
                for i, batch_result in enumerate(batch_results, 1):
                    print(f"  {i}. Query: {batch_result.get('query', 'N/A')}")
                    print(f"     Answer: {batch_result.get('answer', 'N/A')[:100]}...")
                    print(f"     Status: {batch_result.get('status', 'N/A')}")
                    print(f"     Sources: {batch_result.get('source_documents_count', 'N/A')}")
        else:
            print(f"Batch query failed: {response.text}")

    except Exception as e:
        print(f"Batch query error: {e}")


if __name__ == "__main__":
    print("Starting comprehensive MLflow service test...")
    print("=" * 80)

    if test_mlflow_service():
        test_multiple_queries()
        test_batch_queries()
        print("\n" + "=" * 80)
        print("MLflow service test completed successfully!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("MLflow service test failed!")
        print("=" * 80)
