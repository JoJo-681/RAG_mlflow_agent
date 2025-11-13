"""Script to deploy RAG model using MLflow model serving."""

import os
import subprocess
import sys
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.models.mlflow_rag_model import log_rag_model_with_retriever


def deploy_rag_model(model_name: str = "rag_agent", port: int = 5001):
    """
    Deploy RAG model using MLflow model serving

    Args:
        model_name: Name of the model to deploy
        port: Port for the MLflow model server
    """
    try:
        print("Starting RAG model deployment...")

        # Step 1: Log the model to MLflow
        print("Step 1: Logging model to MLflow...")
        run_id = log_rag_model_with_retriever(model_name)

        # Step 2: Start MLflow model serving
        print("Step 2: Starting MLflow model serving...")
        model_uri = f"runs:/{run_id}/{model_name}"

        # Start MLflow model serving in background
        cmd = [
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            model_uri,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
            "--no-conda",
        ]

        print(f"Starting MLflow model server: {' '.join(cmd)}")

        # Start the server process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Give the server time to start
        print(f"Waiting for server to start on port {port}...")
        time.sleep(10)

        # Check if process is still running
        if process.poll() is None:
            print(f"MLflow model server started successfully!")
            print(f"Server running at: http://localhost:{port}")
            print(f"Model URI: {model_uri}")
            print(f"Run ID: {run_id}")
            print("\nUsage examples:")
            print(f"curl -X POST http://localhost:{port}/invocations \\")
            print('  -H "Content-Type: application/json" \\')
            print(
                '  -d \'{"dataframe_split": {"columns": ["question"], "data": [["How do you enable the NPU on the ECU-850b?"]]}}\''
            )

            # Keep the process running
            try:
                process.wait()
            except KeyboardInterrupt:
                print("\nStopping MLflow model server...")
                process.terminate()
                process.wait()
                print("Server stopped.")
        else:
            stdout, stderr = process.communicate()
            print(f"Failed to start MLflow model server:")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            sys.exit(1)

    except (ValueError, RuntimeError, subprocess.SubprocessError) as e:
        print(f"Deployment failed: {e}")
        sys.exit(1)


def test_model_server(port: int = 5001):
    """
    Test the deployed model server

    Args:
        port: Port of the MLflow model server
    """
    import json

    import requests

    url = f"http://localhost:{port}/invocations"

    test_data = {
        "dataframe_split": {
            "columns": ["question"],
            "data": [
                ["What is the maximum operating temperature for the ECU-750?"],
                ["How do you enable the NPU on the ECU-850b?"],
            ],
        }
    }

    try:
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data),
            timeout=30,
        )

        if response.status_code == 200:
            print("Model server test successful!")
            result = response.json()
            print("Response:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Model server test failed: {response.status_code}")
            print(f"Response: {response.text}")

    except (requests.RequestException, TimeoutError, ValueError) as e:
        print(f"Model server test failed: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deploy RAG model using MLflow"
    )
    parser.add_argument("--model-name", default="rag_agent", help="Name of the model")
    parser.add_argument(
        "--port", type=int, default=5001, help="Port for MLflow server"
    )
    parser.add_argument(
        "--test-only", action="store_true", help="Only test existing server"
    )

    args = parser.parse_args()

    if args.test_only:
        test_model_server(args.port)
    else:
        deploy_rag_model(args.model_name, args.port)
