"""
Configuration Management for RAG Application / RAG应用配置管理

This module provides configuration management using Pydantic settings.
All configuration is loaded from environment variables with sensible defaults.
此模块使用Pydantic设置提供配置管理。
所有配置都从环境变量加载，具有合理的默认值。
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings configuration / 应用设置配置

    Settings are loaded from environment variables with the following precedence:
    1. Environment variables
    2. .env file
    3. Default values

    设置按以下优先级从环境变量加载：
    1. 环境变量
    2. .env文件
    3. 默认值
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # LLM Configuration - DeepSeek
    deepseek_api_key: str
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 1000

    # Embedding Configuration - Ollama Local
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_embed: str = "qllama/bge-large-en-v1.5:latest"
    chunk_size: int = 1200
    chunk_overlap: int = 150

    # ChromaDB Configuration
    index_dir: str = "artifacts/chroma_index"
    collection_name: str = "rag_documents"
    data_dir: str = "data"

    # Retrieval Configuration
    top_k: int = 8
    retrieval_score_threshold: float = 0.7

    # MLflow Configuration
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "rag_experiment"
    mlflow_server_port: int = 5000

    # # API Configuration
    # api_host: str = "127.0.0.1"
    # api_port: int = 8000
    # api_title: str = "RAG API with DeepSeek & Ollama"
    # api_version: str = "0.1.0"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance / 获取缓存的设置实例

    Returns:
        Settings: Application configuration instance / 应用配置实例
    """
    return Settings()
