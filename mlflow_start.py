"""
Fixed version of MLflow server startup script for Windows compatibility
"""

import os
import subprocess
import sys
import time


def start_mlflow_server(port=5000, backend_store_uri="./mlruns", default_artifact_root="./mlruns"):
    """
    Start MLflow tracking server with Windows compatibility fixes.

    Args:
        port: Server port number
        backend_store_uri: Backend store URI for MLflow
        default_artifact_root: Default artifact root location
    """
    print(f"Starting MLflow server on port {port}...")

    # Create mlruns directory if it doesn't exist
    os.makedirs(backend_store_uri, exist_ok=True)

    # Windows compatibility fix: use localhost instead of 0.0.0.0
    host = "localhost"

    # Add additional Windows compatibility parameters
    cmd = [
        "mlflow",
        "server",
        "--host",
        host,
        "--port",
        str(port),
        "--backend-store-uri",
        backend_store_uri,
        "--default-artifact-root",
        default_artifact_root,
        # Windows-specific fix parameters
        "--workers",
        "1",  # Limit number of worker processes
    ]

    try:
        print(f"Running command: {' '.join(cmd)}")
        print(f"Windows compatibility fixes applied:")
        print(f"  - Using {host} instead of 0.0.0.0")
        print(f"  - Limiting workers to 1")

        process = subprocess.Popen(cmd)

        print(f"MLflow server starting on http://{host}:{port}")
        print("Waiting for server to initialize...")

        # Wait for server to start
        time.sleep(3)

        # Check if server is running
        try:
            import requests

            response = requests.get(f"http://{host}:{port}", timeout=5)
            if response.status_code == 200:
                print("MLflow server started successfully!")
            else:
                print(f"Server responded with status: {response.status_code}")
        except Exception as e:
            print(f"Could not verify server status: {e}")
            print("But the server process is running...")

        print("Press Ctrl+C to stop the server")

        # Wait for the process
        process.wait()

    except KeyboardInterrupt:
        print("\nStopping MLflow server...")
        process.terminate()
        process.wait()
        print("MLflow server stopped")

    except Exception as e:
        print(f"Error starting MLflow server: {e}")
        print("Troubleshooting tips:")
        print("1. Check if port 5000 is already in use")
        print("2. Try running as administrator")
        print("3. Try a different port (e.g., 5001)")
        sys.exit(1)


def check_mlflow_installation():
    """
    Check if MLflow is installed and available.

    Returns:
        bool: True if MLflow is available, False otherwise
    """
    try:
        subprocess.run(["mlflow", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_port_availability(port=5000):
    """
    Check if the specified port is available.

    Args:
        port: Port number to check

    Returns:
        bool: True if port is available, False otherwise
    """
    import socket

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            return True
    except OSError:
        return False


if __name__ == "__main__":
    # Check if MLflow is installed
    if not check_mlflow_installation():
        print("Error: MLflow is not installed or not in PATH")
        print("Please install MLflow using: pip install mlflow")
        sys.exit(1)

    # Check port availability
    port = 5000
    if not check_port_availability(port):
        print(f"Warning: Port {port} is already in use")
        print("Trying alternative port 5001...")
        port = 5001
        if not check_port_availability(port):
            print(f"Error: Port {port} is also in use")
            print("Please free up the port or specify a different port")
            sys.exit(1)

    # Start MLflow server with Windows compatibility fixes
    start_mlflow_server(port=port)
