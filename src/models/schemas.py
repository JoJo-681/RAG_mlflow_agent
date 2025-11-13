"""Pydantic schemas for API request/response models."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for storing text content and metadata / 用于存储文本内容和元数据的文档模型"""

    content: str = Field(..., description="The text content of the document")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Document metadata (source, author, etc.)"
    )


class DocumentInput(BaseModel):
    """Input model for indexing documents / 用于索引文档的输入模型"""

    documents: List[Document] = Field(..., description="List of documents to index")


class QueryInput(BaseModel):
    """Input model for RAG queries / 用于RAG查询的输入模型"""

    question: str = Field(..., description="The question to answer using RAG")
    k: Optional[int] = Field(None, description="Number of documents to retrieve (overrides config)")
    stream: bool = Field(False, description="Whether to stream the response")


class RetrievedDocument(BaseModel):
    """Model for retrieved documents with scores / 带有分数的检索文档模型"""

    content: str = Field(..., description="Document content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    score: float = Field(..., description="Relevance score")


class QueryResponse(BaseModel):
    """Response model for RAG queries / RAG查询的响应模型"""

    answer: str = Field(..., description="Generated answer")
    retrieved_documents: List[RetrievedDocument] = Field(
        ..., description="Documents retrieved for context"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata (latency, token count, etc.)"
    )


class IndexResponse(BaseModel):
    """Response model for document indexing / 文档索引的响应模型"""

    status: str = Field(..., description="Status of the indexing operation")
    indexed_count: int = Field(..., description="Number of documents indexed")
    collection_name: str = Field(..., description="Name of the vector collection")


class FileUploadResponse(BaseModel):
    """Response model for file upload / 文件上传的响应模型"""

    status: str = Field(..., description="Status of the upload operation")
    filename: str = Field(..., description="Name of the uploaded file")
    indexed_count: int = Field(..., description="Number of chunks indexed")
    file_size: int = Field(..., description="Size of the uploaded file in bytes")
    message: str = Field(..., description="Additional information about the upload")


class HealthResponse(BaseModel):
    """Health check response model / 健康检查响应模型"""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(
        default_factory=dict, description="Status of individual components"
    )
