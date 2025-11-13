"""MLflow tracking utilities for RAG experiments / 用于RAG实验的MLflow跟踪工具"""

import time
from functools import wraps
from typing import Any, Dict, Optional

import mlflow

from src.config import get_settings


class MLflowTracker:
    """MLflow tracker for RAG experiments and metrics / MLflow跟踪器用于RAG实验和指标"""

    def __init__(self):
        """Initialize MLflow tracker / 初始化MLflow跟踪器"""
        self.settings = get_settings()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)

        # Set or create experiment
        try:
            mlflow.set_experiment(self.settings.mlflow_experiment_name)
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Warning: Could not set MLflow experiment: {e}")

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run / 开始新的MLflow运行"""
        return mlflow.start_run(run_name=run_name)

    def end_run(self):
        """End the current MLflow run / 结束当前MLflow运行"""
        mlflow.end_run()

    def log_query(self, question: str, answer: str, retrieved_docs: list, metadata: Dict[str, Any]):
        """
        Log a RAG query to MLflow / 将RAG查询记录到MLflow

        Args:
            question: The input question / 输入问题
            answer: The generated answer / 生成的答案
            retrieved_docs: List of retrieved documents / 检索到的文档列表
            metadata: Additional metadata (timing, scores, etc.) / 额外的元数据（时间、分数等）
        """
        try:
            with mlflow.start_run(nested=True):
                # Log parameters
                mlflow.log_param("question_length", len(question))
                mlflow.log_param("num_retrieved_docs", len(retrieved_docs))
                mlflow.log_param("model", self.settings.deepseek_model)
                mlflow.log_param("embedding_model", self.settings.ollama_embed)

                # Log metrics
                mlflow.log_metric("answer_length", len(answer))

                if "retrieval_time" in metadata:
                    mlflow.log_metric("retrieval_time_seconds", metadata["retrieval_time"])

                if "generation_time" in metadata:
                    mlflow.log_metric("generation_time_seconds", metadata["generation_time"])

                if "total_time" in metadata:
                    mlflow.log_metric("total_time_seconds", metadata["total_time"])

                # Log average relevance score
                if retrieved_docs:
                    avg_score = sum(doc.get("score", 0) for doc in retrieved_docs) / len(
                        retrieved_docs
                    )
                    mlflow.log_metric("avg_relevance_score", avg_score)

                # Log as artifacts
                mlflow.log_text(question, "question.txt")
                mlflow.log_text(answer, "answer.txt")

                # Log metadata as JSON
                mlflow.log_dict(metadata, "metadata.json")

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error logging to MLflow: {e}")

    def log_indexing(
        self, num_documents: int, num_chunks: int, collection_name: str, duration: float
    ):
        """
        Log document indexing information / 记录文档索引信息

        Args:
            num_documents: Number of source documents / 源文档数量
            num_chunks: Number of chunks created / 创建的块数量
            collection_name: Name of the vector collection / 向量集合名称
            duration: Indexing duration in seconds / 索引持续时间（秒）
        """
        try:
            with mlflow.start_run(run_name="indexing", nested=True):
                mlflow.log_param("collection_name", collection_name)
                mlflow.log_metric("num_documents", num_documents)
                mlflow.log_metric("num_chunks", num_chunks)
                mlflow.log_metric("indexing_duration_seconds", duration)
                mlflow.log_metric("chunks_per_document", num_chunks / max(num_documents, 1))
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error logging indexing to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, float]):
        """Log multiple metrics at once / 一次性记录多个指标"""
        try:
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error logging metrics to MLflow: {e}")

    def log_params(self, params: Dict[str, Any]):
        """Log multiple parameters at once / 一次性记录多个参数"""
        try:
            for name, value in params.items():
                mlflow.log_param(name, value)
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error logging params to MLflow: {e}")


def track_query(func):
    """Decorator to automatically track RAG queries with MLflow / 装饰器自动使用MLflow跟踪RAG查询"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        tracker = MLflowTracker()
        start_time = time.time()

        # Execute the function
        result = func(*args, **kwargs)

        # Calculate duration
        duration = time.time() - start_time

        # Log to MLflow if result contains expected fields
        if isinstance(result, dict) and "answer" in result:
            metadata = result.get("metadata", {})
            metadata["function_duration"] = duration

            tracker.log_query(
                question=kwargs.get("question", ""),
                answer=result["answer"],
                retrieved_docs=result.get("retrieved_documents", []),
                metadata=metadata,
            )

        return result

    return wrapper
