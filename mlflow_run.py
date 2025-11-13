"""Run MLflow RAG model logging with ChromaDB."""

import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.config import get_settings
from src.models.mlflow_rag_model import log_rag_model_with_retriever


def main():
    """Main function to run MLflow RAG model logging."""
    settings = get_settings()

    print("Starting RAG System with MLflow")
    print("MLflow UI: http://localhost:5000")
    print(f"LLM: {settings.deepseek_model}")
    print(f"Embeddings: {settings.ollama_embed}")
    print(f"ChromaDB Index: {settings.index_dir}")
    print()

    print("Logging RAG model to MLflow...")
    try:
        run_id = log_rag_model_with_retriever("rag_agent")
        print(f"Model logged successfully! Run ID: {run_id}")
        print("View in MLflow UI: http://localhost:5000")
        print(f"Model URI: runs:/{run_id}/rag_agent")
    except (ValueError, RuntimeError, ImportError) as e:
        print(f"Error logging model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
