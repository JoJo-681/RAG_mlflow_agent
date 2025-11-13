"""ChromaDB retriever for document retrieval / 用于文档检索的ChromaDB检索器"""

import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import get_settings


class ChromaRetriever:
    """Manages document indexing and retrieval using ChromaDB / 使用ChromaDB管理文档索引和检索"""

    def __init__(self):
        """Initialize the ChromaDB retriever with Ollama embeddings / 使用Ollama嵌入初始化ChromaDB检索器"""
        self.settings = get_settings()

        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(
            model=self.settings.ollama_embed, base_url=self.settings.ollama_base_url
        )

        # Initialize text splitter for markdown
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

        # Create persist directory if it doesn't exist
        os.makedirs(self.settings.index_dir, exist_ok=True)

        # Initialize or load the vector store
        self.vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.index_dir,
        )

    def load_markdown_files(self, directory: str = None) -> List[Document]:
        """
        Load all markdown files from a directory / 从目录加载所有markdown文件

        Args:
            directory: Directory containing markdown files (uses config default if None) / 包含markdown文件的目录（如果为None则使用配置默认值）

        Returns:
            List of loaded documents / 加载的文档列表
        """
        directory = directory or self.settings.data_dir
        markdown_files = list(Path(directory).glob("*.md"))

        all_docs = []
        for file_path in markdown_files:
            try:
                # Use TextLoader as fallback to avoid unstructured dependency issues
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

                # Add source file to metadata
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["filename"] = file_path.name
                    doc.metadata["file_type"] = ".md"

                all_docs.extend(docs)
                print(f"Loaded: {file_path.name}")
            except (ValueError, FileNotFoundError, UnicodeDecodeError) as e:
                print(f"Error loading {file_path.name}: {e}")

        return all_docs

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file based on its extension / 根据文件扩展名加载单个文件

        Args:
            file_path: Path to the file to load / 要加载的文件路径

        Returns:
            List of loaded documents / 加载的文档列表
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()

        try:
            # Choose appropriate loader based on file extension
            if file_extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))
            elif file_extension == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_extension in [".txt", ".text"]:
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                # Use unstructured loader for other file types
                loader = UnstructuredFileLoader(str(file_path))

            docs = loader.load()

            # Add source file to metadata
            for doc in docs:
                doc.metadata["source"] = str(file_path)
                doc.metadata["filename"] = file_path.name
                doc.metadata["file_type"] = file_extension

            print(f"Loaded: {file_path.name} ({file_extension})")
            return docs

        except (ValueError, FileNotFoundError, UnicodeDecodeError) as e:
            print(f"Error loading {file_path.name}: {e}")
            raise

    def delete_documents_by_filename(self, filename: str) -> int:
        """
        Delete all documents with the specified filename from the vector store / 从向量存储中删除具有指定文件名的所有文档

        Args:
            filename: The filename to delete documents for / 要删除文档的文件名

        Returns:
            Number of documents deleted / 删除的文档数量
        """
        try:
            # Get all documents from the collection
            all_docs = self.vectorstore.get()

            if not all_docs or "metadatas" not in all_docs:
                print(f"No documents found in collection")
                return 0

            # Find document IDs to delete
            ids_to_delete = []
            for idx, metadata in enumerate(all_docs["metadatas"]):
                if metadata and metadata.get("filename") == filename:
                    ids_to_delete.append(all_docs["ids"][idx])

            if not ids_to_delete:
                print(f"No documents found for filename: {filename}")
                return 0

            # Delete the documents
            self.vectorstore.delete(ids=ids_to_delete)

            print(f"Deleted {len(ids_to_delete)} documents for filename: {filename}")
            return len(ids_to_delete)

        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Error deleting documents for filename {filename}: {e}")
            return 0

    def index_file(self, file_path: str, delete_existing: bool = True) -> int:
        """
        Load and index a single file / 加载并索引单个文件

        Args:
            file_path: Path to the file to index / 要索引的文件路径
            delete_existing: Whether to delete existing chunks for this file first / 是否先删除此文件的现有块

        Returns:
            Number of chunks indexed / 索引的块数量
        """
        file_path = Path(file_path)
        filename = file_path.name

        # Delete existing chunks for this file if requested
        if delete_existing:
            deleted_count = self.delete_documents_by_filename(filename)
            if deleted_count > 0:
                print(f"Deleted {deleted_count} existing chunks for {filename}")

        # Load the file
        docs = self.load_file(str(file_path))

        if not docs:
            print(f"No content found in file: {file_path}")
            return 0

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(docs)

        # Add chunk IDs to metadata
        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = idx

        # Add to vector store (newer versions auto-persist)
        self.vectorstore.add_documents(split_docs)

        print(f"Indexed {len(split_docs)} chunks from {file_path}")
        return len(split_docs)

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the vector store / 将文档添加到向量存储

        Args:
            documents: List of documents with 'content' and 'metadata' fields / 包含'content'和'metadata'字段的文档列表

        Returns:
            Number of chunks indexed / 索引的块数量
        """
        # Convert to LangChain Document format
        docs = []
        for doc in documents:
            # Split the document into chunks
            chunks = self.text_splitter.split_text(doc["content"])

            # Create a Document for each chunk
            for idx, chunk in enumerate(chunks):
                metadata = doc.get("metadata", {}).copy()
                metadata["chunk_id"] = idx
                docs.append(Document(page_content=chunk, metadata=metadata))

        # Add to vector store (newer versions auto-persist)
        if docs:
            self.vectorstore.add_documents(docs)

        return len(docs)

    def index_markdown_files(self, directory: str = None, delete_existing: bool = True) -> int:
        """
        Load and index all markdown files from a directory / 从目录加载并索引所有markdown文件

        Args:
            directory: Directory containing markdown files / 包含markdown文件的目录
            delete_existing: Whether to delete existing chunks for files before indexing / 是否在索引前删除文件的现有块

        Returns:
            Number of chunks indexed / 索引的块数量
        """
        # Load markdown files
        docs = self.load_markdown_files(directory)

        if not docs:
            print("No markdown files found to index")
            return 0

        # Get unique filenames from loaded documents
        unique_filenames = set()
        for doc in docs:
            filename = doc.metadata.get("filename")
            if filename:
                unique_filenames.add(filename)

        # Delete existing chunks for these files if requested
        if delete_existing and unique_filenames:
            total_deleted = 0
            for filename in unique_filenames:
                deleted_count = self.delete_documents_by_filename(filename)
                total_deleted += deleted_count
            if total_deleted > 0:
                print(f"Deleted {total_deleted} existing chunks for {len(unique_filenames)} files")

        # Split documents into chunks
        split_docs = self.text_splitter.split_documents(docs)

        # Add chunk IDs to metadata
        for idx, doc in enumerate(split_docs):
            doc.metadata["chunk_id"] = idx

        # Add to vector store (newer versions auto-persist)
        self.vectorstore.add_documents(split_docs)

        print(f"Indexed {len(split_docs)} chunks from {len(docs)} markdown files")
        return len(split_docs)

    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query / 为查询检索相关文档

        Args:
            query: The search query / 搜索查询
            k: Number of documents to retrieve (uses config default if None) / 要检索的文档数量（如果为None则使用配置默认值）

        Returns:
            List of retrieved documents with content, metadata, and scores / 包含内容、元数据和分数的检索文档列表
        """
        k = k or self.settings.top_k

        # Perform similarity search with scores
        results = self.vectorstore.similarity_search_with_score(query, k=k)

        # Format results
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(1 - score),  # Convert distance to similarity
                }
            )

        return retrieved_docs

    def clear_collection(self) -> None:
        """Clear all documents from the collection / 清除集合中的所有文档"""
        # Delete and recreate the collection
        self.vectorstore.delete_collection()
        self.vectorstore = Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.settings.index_dir,
        )

    def check_vector_store_exists(self) -> bool:
        """
        Check if vector store exists and has documents / 检查向量存储是否存在且包含文档

        Returns:
            bool: True if vector store exists and has documents, False otherwise
        """
        try:
            count = self.get_collection_count()
            print(f"Vector store check: {count} documents found")
            return count > 0
        except Exception as e:
            print(f"Vector store check failed: {e}")
            return False

    def ensure_vector_store_exists(self, auto_create: bool = True) -> bool:
        """
        Ensure vector store exists, create if needed / 确保向量存储存在，必要时创建

        Args:
            auto_create: Whether to automatically create vector store if it doesn't exist

        Returns:
            bool: True if vector store exists or was created successfully
        """
        print("Checking vector store status...")

        # Check if vector store exists and has documents
        if self.check_vector_store_exists():
            print("Vector store exists and contains documents")
            return True

        print("Vector store is empty or doesn't exist")

        if auto_create:
            print("Auto-creating vector store with default documents...")
            try:
                # Index default data directory
                chunks_count = self.index_markdown_files(delete_existing=False)
                if chunks_count > 0:
                    print(f"Successfully created vector store with {chunks_count} chunks")
                    return True
                else:
                    print("Failed to create vector store - no documents were indexed")
                    return False
            except Exception as e:
                print(f"Error auto-creating vector store: {e}")
                return False
        else:
            print("Vector store auto-creation is disabled")
            return False

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection / 获取集合中的文档数量"""
        return self.vectorstore._collection.count()
