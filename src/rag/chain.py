"""LangChain RAG chain implementation with DeepSeek / 使用DeepSeek的LangChain RAG链实现"""

from typing import Any, Dict, List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import get_settings
from src.rag.retriever import ChromaRetriever


class RAGChain:
    """RAG chain using LangChain with DeepSeek LLM / 使用LangChain和DeepSeek LLM的RAG链"""

    def __init__(self, retriever: ChromaRetriever = None):
        """Initialize the RAG chain / 初始化RAG链"""
        self.settings = get_settings()
        self.retriever = retriever or ChromaRetriever()

        # Initialize DeepSeek LLM using OpenAI-compatible API
        self.llm = ChatOpenAI(
            model=self.settings.deepseek_model,
            openai_api_key=self.settings.deepseek_api_key,
            openai_api_base=self.settings.deepseek_base_url,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )

        # Create the RAG prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Provide concise, direct answers based on the provided context.

IMPORTANT INSTRUCTIONS:
1. Provide direct, concise answers without unnecessary formatting or explanations
2. Do NOT use markdown formatting (no **bold**, *italic*, tables, or code blocks)
3. Do NOT include source citations in the answer text
4. Keep answers brief and to the point - aim for 1-3 sentences maximum
5. Focus on the key information requested in the question
6. If comparing multiple items, use simple bullet points or short phrases
7. Do NOT repeat the question or add introductory phrases

Context:
{context}""",
                ),
                ("user", "{question}"),
            ]
        )

        # Build the RAG chain
        self.chain = (
            {
                "context": lambda x: self._format_docs(
                    self.retriever.retrieve(x["question"], k=x.get("k"))
                ),
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents for the prompt / 格式化检索到的文档用于提示"""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.get("metadata", {}).get("filename", "Unknown")
            content = doc.get("content", "")
            formatted.append(f"[Source {i}: {source}]\n{content}\n")
        return "\n".join(formatted)

    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Query the RAG chain / 查询RAG链

        Args:
            question: The question to answer / 要回答的问题
            k: Number of documents to retrieve / 要检索的文档数量

        Returns:
            Dictionary with answer and retrieved documents / 包含答案和检索文档的字典
        """
        # Retrieve documents
        retrieved_docs = self.retriever.retrieve(question, k=k)

        # Generate answer
        answer = self.chain.invoke({"question": question, "k": k})

        return {"answer": answer, "retrieved_documents": retrieved_docs}

    def stream_query(self, question: str, k: int = None):
        """
        Stream the RAG chain response / 流式传输RAG链响应

        Args:
            question: The question to answer / 要回答的问题
            k: Number of documents to retrieve / 要检索的文档数量

        Yields:
            Chunks of the generated answer / 生成的答案块
        """
        # First, retrieve documents (not streamed)
        retrieved_docs = self.retriever.retrieve(question, k=k)

        # Stream the answer generation
        for chunk in self.chain.stream({"question": question, "k": k}):
            yield {"chunk": chunk, "retrieved_documents": retrieved_docs if chunk else None}
