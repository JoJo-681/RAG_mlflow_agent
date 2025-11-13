"""Deploy RAG model with MLflow serving and API exposure. / 使用MLflow服务和API暴露部署RAG模型"""

import os
import subprocess
import sys
import time

import requests

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import mlflow

from src.config import get_settings
from src.models.mlflow_rag_model import log_rag_model_with_retriever


def start_mlflow_tracking_server():
    """Start MLflow tracking server for UI. / 启动MLflow跟踪服务器用于UI"""
    settings = get_settings()
    print(f"Starting MLflow Tracking Server on {settings.mlflow_tracking_uri}")

    # Check if server is already running / 检查服务器是否已在运行
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.search_experiments()
        print("MLflow tracking server is already running")
        return True
    except:
        print("Starting MLflow tracking server...")

        # Start MLflow tracking server / 启动MLflow跟踪服务器
        process = subprocess.Popen(
            [
                "mlflow",
                "server",
                "--host",
                "0.0.0.0",
                "--port",
                "5000",
                "--backend-store-uri",
                "./mlruns",
                "--default-artifact-root",
                "./mlruns",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start / 等待服务器启动
        time.sleep(5)

        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            mlflow.search_experiments()
            print("MLflow tracking server started successfully")
            return True
        except:
            print("Failed to start MLflow tracking server")
            return False


def deploy_rag_model():
    """Deploy RAG model to MLflow Model Registry. / 将RAG模型部署到MLflow模型注册表"""
    get_settings()

    print("Deploying RAG model to MLflow...")

    # Log the model / 记录模型
    run_id = log_rag_model_with_retriever("rag_agent_deployed")

    # Get model URI / 获取模型URI
    model_uri = f"runs:/{run_id}/rag_agent_deployed"
    print(f"Model deployed successfully: {model_uri}")

    return model_uri


def start_mlflow_model_serving(model_uri, port=5001):
    """Start MLflow model serving with REST API. / 启动MLflow模型服务与REST API"""
    print(f"Starting MLflow Model Serving on port {port}...")

    # Start MLflow model serving / 启动MLflow模型服务
    process = subprocess.Popen(
        [
            "mlflow",
            "models",
            "serve",
            "--model-uri",
            model_uri,
            "--port",
            str(port),
            "--host",
            "0.0.0.0",
            "--no-conda",  # Add this flag to avoid conda environment issues / 添加此标志以避免conda环境问题
            "--env-manager",
            "local",  # Use local environment instead of creating virtual env / 使用本地环境而不是创建虚拟环境
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to start with retry logic / 使用重试逻辑等待服务器启动
    print("Waiting for model serving to start...")
    max_retries = 5
    for i in range(max_retries):
        time.sleep(5)  # Wait 5 seconds between retries / 重试之间等待5秒
        try:
            response = requests.get(f"http://localhost:{port}/ping", timeout=10)
            if response.status_code == 200:
                print(f"MLflow Model Serving started successfully on port {port}")
                return process, port
            else:
                print(f"Health check attempt {i+1}/{max_retries}: Status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"Health check attempt {i+1}/{max_retries}: Connection refused")
        except requests.exceptions.Timeout:
            print(f"Health check attempt {i+1}/{max_retries}: Timeout")
        except Exception as e:
            print(f"Health check attempt {i+1}/{max_retries}: {e}")

    print("Model serving health check failed after all retries")
    if process:
        process.terminate()
    return None, None


def test_model_api(port=5001):
    """Test the model API with curl-like requests. / 使用类似curl的请求测试模型API"""
    print(f"Testing model API on port {port}...")

    # Test data / 测试数据
    test_queries = [
        "What is the maximum operating temperature?",
        "What is the power consumption of the ECU-850b under load?",
        "Which ECU models support Over-the-Air (OTA) updates?",
    ]

    for query in test_queries:
        try:
            # Prepare request data / 准备请求数据
            data = {"dataframe_split": {"columns": ["query"], "data": [[query]]}}

            # Make prediction request / 进行预测请求
            response = requests.post(
                f"http://localhost:{port}/invocations",
                json=data,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Query: '{query}'")

                # Handle different response formats / 处理不同的响应格式
                if "predictions" in result:
                    prediction = result["predictions"][0]
                    if isinstance(prediction, dict):
                        print(f"   Answer: {prediction.get('answer', 'N/A')[:100]}...")
                        print(f"   Status: {prediction.get('status', 'N/A')}")
                        print(f"   Sources: {prediction.get('source_documents_count', 'N/A')}")
                    else:
                        print(f"   Raw prediction: {prediction}")
                else:
                    print(f"   Raw response: {result}")
                print()
            else:
                print(f"Failed for query: '{query}' - Status: {response.status_code}")
                print(f"   Response: {response.text}")

        except Exception as e:
            print(f"Error testing query '{query}': {e}")
            print(f"   Response content: {response.text if 'response' in locals() else 'No response'}")


def main():
    """Main deployment function. / 主要部署函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy RAG model using MLflow")
    parser.add_argument("--model-name", default="rag_agent_deployed", help="Name of the model")
    parser.add_argument("--port", type=int, default=5001, help="Port for MLflow model server")
    parser.add_argument("--test-only", action="store_true", help="Only test existing server")
    
    args = parser.parse_args()

    if args.test_only:
        print("Testing existing model server...")
        test_model_api(args.port)
        return

    print("=" * 60)
    print("MLflow RAG Model Deployment")
    print("=" * 60)

    # Step 1: Start MLflow tracking server (for UI) / 步骤1: 启动MLflow跟踪服务器(用于UI)
    if not start_mlflow_tracking_server():
        return

    # Step 2: Deploy model to MLflow / 步骤2: 将模型部署到MLflow
    model_uri = deploy_rag_model()

    # Step 3: Start MLflow model serving / 步骤3: 启动MLflow模型服务
    serving_process, serving_port = start_mlflow_model_serving(model_uri, port=args.port)

    if serving_process:
        print("\n" + "=" * 60)
        print("Deployment Complete!")
        print("=" * 60)
        print(f"MLflow UI: http://localhost:5000")
        print(f"Model API: http://localhost:{serving_port}")
        print(f"Model URI: {model_uri}")
        print(f"Model Name: {args.model_name}")
        print("\nUsage Examples:")
        print("1. View experiments in browser: http://localhost:5000")
        print("2. Test API with curl:")
        print(f"   curl -X POST http://localhost:{serving_port}/invocations \\")
        print('        -H "Content-Type: application/json" \\')
        print('        -d \'{"dataframe_split": {"columns": ["query"], "data": [["What is the maximum temperature?"]]}}\'')
        print("\n3. Test API with Python requests:")
        print("   import requests")
        print('   data = {"dataframe_split": {"columns": ["query"], "data": [["Your question here"]]}}')
        print(f'   response = requests.post("http://localhost:{serving_port}/invocations", json=data)')

        print("\nRunning API tests...")
        test_model_api(serving_port)

        print("\nTo stop the servers:")
        print("   - Press Ctrl+C in this terminal")
        print("   - Or manually kill the processes")

        try:
            # Keep the servers running
            serving_process.wait()
        except KeyboardInterrupt:
            print("\nStopping servers...")
            serving_process.terminate()
    else:
        print("Deployment failed")


if __name__ == "__main__":
    main()
