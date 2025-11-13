"""LangGraph workflow for RAG pipeline / 用于RAG管道的LangGraph工作流"""

import time
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from src.rag.chain import RAGChain
from src.rag.retriever import ChromaRetriever


class RAGState(TypedDict):
    """State for the RAG workflow / RAG工作流的状态"""

    question: str
    k: int
    retrieved_documents: List[Dict[str, Any]]
    answer: str
    metadata: Dict[str, Any]
    error: str


def retrieve_node(state: RAGState) -> RAGState:
    """Retrieve relevant documents / 检索相关文档"""
    try:
        start_time = time.time()
        retriever = ChromaRetriever()

        # Retrieve documents
        retrieved_docs = retriever.retrieve(query=state["question"], k=state.get("k"))

        retrieval_time = time.time() - start_time

        # Update state
        state["retrieved_documents"] = retrieved_docs
        state["metadata"] = state.get("metadata", {})
        state["metadata"]["retrieval_time"] = retrieval_time
        state["metadata"]["num_retrieved"] = len(retrieved_docs)

    except (ValueError, RuntimeError, ImportError) as e:
        state["error"] = f"Retrieval error: {str(e)}"

    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate answer using LLM / 使用LLM生成答案"""
    try:
        # Check if retrieval was successful
        if state.get("error"):
            return state

        start_time = time.time()

        # Use RAG chain to generate answer
        rag_chain = RAGChain()
        result = rag_chain.query(question=state["question"], k=state.get("k"))

        generation_time = time.time() - start_time

        # Update state
        state["answer"] = result["answer"]
        state["metadata"]["generation_time"] = generation_time
        state["metadata"]["total_time"] = (
            state["metadata"].get("retrieval_time", 0) + generation_time
        )

    except (ValueError, RuntimeError, ImportError) as e:
        state["error"] = f"Generation error: {str(e)}"

    return state


def should_continue(state: RAGState) -> str:
    """Determine if the workflow should continue / 确定工作流是否应该继续"""
    if state.get("error"):
        return "end"
    return "generate"


def create_rag_graph() -> StateGraph:
    """
    Create the RAG workflow graph using LangGraph / 使用LangGraph创建RAG工作流图

    Returns:
        Compiled StateGraph workflow / 编译的StateGraph工作流
    """
    # Create the graph
    workflow = StateGraph(RAGState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    # Set entry point
    workflow.set_entry_point("retrieve")

    # Add edges
    workflow.add_conditional_edges(
        "retrieve", should_continue, {"generate": "generate", "end": END}
    )

    workflow.add_edge("generate", END)

    # Compile the graph
    return workflow.compile()


class RAGWorkflow:
    """Wrapper class for RAG workflow execution / RAG工作流执行的包装类"""

    def __init__(self):
        """Initialize the RAG workflow / 初始化RAG工作流"""
        self.graph = create_rag_graph()

    def run(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Execute the RAG workflow / 执行RAG工作流

        Args:
            question: The question to answer / 要回答的问题
            k: Number of documents to retrieve / 要检索的文档数量

        Returns:
            Dictionary with answer, retrieved documents, and metadata / 包含答案、检索文档和元数据的字典
        """
        # Initialize state
        initial_state: RAGState = {
            "question": question,
            "k": k,
            "retrieved_documents": [],
            "answer": "",
            "metadata": {},
            "error": "",
        }

        # Run the workflow
        result = self.graph.invoke(initial_state)

        # Check for errors
        if result.get("error"):
            return {
                "answer": f"Error: {result['error']}",
                "retrieved_documents": [],
                "metadata": result.get("metadata", {}),
            }

        # Return the result
        return {
            "answer": result["answer"],
            "retrieved_documents": result["retrieved_documents"],
            "metadata": result["metadata"],
        }
