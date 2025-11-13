"""
MLflow RAG Model Implementation / MLflow RAG模型实现

This module provides a MLflow PythonModel implementation for a Retrieval-Augmented
Generation (RAG) system. The model integrates ChromaDB for vector storage and
OpenAI-compatible LLMs for text generation.
此模块提供了检索增强生成(RAG)系统的MLflow PythonModel实现。
该模型集成了ChromaDB用于向量存储和OpenAI兼容的LLM用于文本生成。
"""

from typing import Any, Dict

import mlflow
import mlflow.pyfunc
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema

from src.config import get_settings


class RAGMLflowModel(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel implementation for RAG system / RAG系统的MLflow PythonModel实现

    This model handles document retrieval and generation using ChromaDB
    vector storage and OpenAI-compatible language models.
    该模型使用ChromaDB向量存储和OpenAI兼容的语言模型处理文档检索和生成。
    """

    def __init__(self):
        """Initialize the RAG MLflow model with configuration settings / 使用配置设置初始化RAG MLflow模型"""
        self.settings = get_settings()

    def load_context(self, context):
        """
        Load model context when the model is loaded / 加载模型时加载模型上下文

        This method is called automatically by MLflow when the model
        is loaded for serving.
        当模型加载用于服务时，此方法由MLflow自动调用。
        """

    def _initialize_components(self):
        """
        Initialize RAG components on demand / 按需初始化RAG组件

        Returns:
            tuple: (vectorstore, llm) - ChromaDB vector store and LLM instance / ChromaDB向量存储和LLM实例
        """
        embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed, base_url=self.settings.ollama_base_url
        )

        vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=embeddings,
            persist_directory=self.settings.index_dir,
        )

        # Check if vector store exists and has documents
        try:
            doc_count = vectorstore._collection.count()
            if doc_count == 0:
                print(f"Warning: Vector store exists but contains {doc_count} documents")
                print("Please run init_vector_db.py to populate the vector store")
        except Exception as e:
            print(f"Error checking vector store: {e}")
            print("Vector store may not be properly initialized")

        llm = ChatOpenAI(
            model=self.settings.deepseek_model,
            openai_api_key=self.settings.deepseek_api_key,
            openai_api_base=self.settings.deepseek_base_url,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        return vectorstore, llm

    def predict(self, model_input) -> Dict[str, Any]:
        """
        Predict method for MLflow model serving / MLflow模型服务的预测方法

        This method handles different input formats and processes queries
        through the RAG pipeline.
        此方法处理不同的输入格式，并通过RAG管道处理查询。

        Args:
            model_input: Input query as string, list of strings, or DataFrame / 输入查询，可以是字符串、字符串列表或DataFrame

        Returns:
            Dictionary containing answer and metadata / 包含答案和元数据的字典
        """
        if isinstance(model_input, str):
            return self._process_single_query(model_input)

        elif isinstance(model_input, list) and all(isinstance(item, str) for item in model_input):
            if len(model_input) == 1:
                return self._process_single_query(model_input[0])
            else:
                return self._process_batch_queries(model_input)

        elif isinstance(model_input, pd.DataFrame):
            if "query" not in model_input.columns:
                raise ValueError("Input DataFrame must contain 'query' column")
            if len(model_input) == 1:
                return self._process_single_query(model_input.iloc[0]["query"])
            else:
                return self._process_batch_queries(model_input["query"].tolist())

        else:
            raise ValueError("model_input must be a string, list of strings, or pandas DataFrame")

    def _process_single_query(self, query: str) -> Dict[str, Any]:
        """
        Process a single query through the RAG system / 通过RAG系统处理单个查询

        This method retrieves relevant documents and generates an answer
        using the language model.
        此方法检索相关文档并使用语言模型生成答案。

        Args:
            query: Input query string / 输入查询字符串

        Returns:
            Dictionary with answer, source document count, and status / 包含答案、源文档数量和状态的字典
        """
        try:
            vectorstore, llm = self._initialize_components()

            retriever = vectorstore.as_retriever()
            docs = retriever.invoke(query)

            context = "\n\n".join([doc.page_content for doc in docs])

            prompt_template = (
                "Use the following pieces of context to answer the question at the end. "
                "If you don't know the answer, just say that you don't know, "
                "don't try to make up an answer.\n\nContext:\n{context}\n\n"
                "Question: {question}\nAnswer:"
            )

            response = llm.invoke(prompt_template.format(context=context, question=query))

            return {
                "query": query,
                "answer": response.content,
                "source_documents_count": len(docs),
                "status": "success",
            }
        except (ValueError, RuntimeError, ImportError) as e:
            return {
                "query": query,
                "answer": "",
                "source_documents_count": 0,
                "status": f"error: {str(e)}",
            }

    def _process_batch_queries(self, queries: list) -> Dict[str, Any]:
        """
        Process multiple queries through the RAG system / 通过RAG系统处理多个查询

        This method processes a batch of queries and returns results for all.
        此方法处理一批查询并返回所有结果。

        Args:
            queries: List of query strings / 查询字符串列表

        Returns:
            Dictionary with batch results containing answers for all queries / 包含所有查询答案的批量结果字典
        """
        try:
            vectorstore, llm = self._initialize_components()
            retriever = vectorstore.as_retriever()

            batch_results = []

            for query in queries:
                try:
                    docs = retriever.invoke(query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    prompt_template = (
                        "Use the following pieces of context to answer the question at the end. "
                        "If you don't know the answer, just say that you don't know, "
                        "don't try to make up an answer.\n\nContext:\n{context}\n\n"
                        "Question: {question}\nAnswer:"
                    )

                    response = llm.invoke(prompt_template.format(context=context, question=query))

                    batch_results.append(
                        {
                            "query": query,
                            "answer": response.content,
                            "source_documents_count": len(docs),
                            "status": "success",
                        }
                    )
                except Exception as e:
                    batch_results.append(
                        {
                            "query": query,
                            "answer": "",
                            "source_documents_count": 0,
                            "status": f"error: {str(e)}",
                        }
                    )

            return {
                "batch_results": batch_results,
                "total_queries": len(queries),
                "successful_queries": sum(1 for r in batch_results if r["status"] == "success"),
                "failed_queries": sum(1 for r in batch_results if r["status"] != "success"),
                "status": "batch_completed",
            }
        except Exception as e:
            return {
                "batch_results": [],
                "total_queries": len(queries),
                "successful_queries": 0,
                "failed_queries": len(queries),
                "status": f"batch_error: {str(e)}",
            }


def load_retriever(persist_directory: str):
    """
    Load retriever function for MLflow model logging / 用于MLflow模型记录的检索器加载函数

    Args:
        persist_directory: Directory where ChromaDB is persisted / ChromaDB持久化目录

    Returns:
        ChromaDB retriever instance / ChromaDB检索器实例
    """
    embeddings = OllamaEmbeddings(
        model=get_settings().ollama_embed, base_url=get_settings().ollama_base_url
    )

    vectorstore = Chroma(
        collection_name=get_settings().collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

    return vectorstore.as_retriever()


def log_rag_model_with_retriever(model_name: str = "rag_agent"):
    """
    Log the RAG model to MLflow using the retriever loader function pattern / 使用检索器加载器函数模式将RAG模型记录到MLflow

    Args:
        model_name: Name for the logged model / 记录模型的名称

    Returns:
        Run ID of the logged model / 记录模型的运行ID
    """
    settings = get_settings()
    mlflow.set_experiment(settings.mlflow_experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_deployment") as run:
        mlflow.log_params(
            {
                "model_type": "rag_agent",
                "llm_model": settings.deepseek_model,
                "embedding_model": settings.ollama_embed,
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "collection_name": settings.collection_name,
            }
        )

        input_example = pd.DataFrame(
            {"query": ["What is the maximum operating temperature for the ECU-750?"]}
        )

        input_schema = Schema([ColSpec("string", "query")])
        output_schema = Schema(
            [
                ColSpec("string", "query"),
                ColSpec("string", "answer"),
                ColSpec("double", "source_documents_count"),
                ColSpec("string", "status"),
            ]
        )

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        model_info = mlflow.pyfunc.log_model(
            python_model=RAGMLflowModel(),
            artifact_path=model_name,
            input_example=input_example,
            signature=signature,
        )

        mlflow.log_params({"run_id": run.info.run_id, "model_uri": model_info.model_uri})

        print(f"Model logged successfully. Run ID: {run.info.run_id}")
        print(f"Model URI: {model_info.model_uri}")

        return run.info.run_id


def load_rag_model(model_uri: str):
    """
    Load a logged RAG model from MLflow / 从MLflow加载已记录的RAG模型

    Args:
        model_uri: MLflow model URI / MLflow模型URI

    Returns:
        Loaded RAG model instance / 加载的RAG模型实例
    """
    return mlflow.pyfunc.load_model(model_uri)


if __name__ == "__main__":
    run_id = log_rag_model_with_retriever("rag_agent_v1")
    print(f"Model logged with run_id: {run_id}")
