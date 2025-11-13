#!/usr/bin/env python3
"""
Script for initializing and managing the vector database
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import get_settings
from src.rag.retriever import ChromaRetriever


def initialize_vector_db(
    clear_existing=False, file_path=None, directory=None, delete_by_filename=None
):
    """
    Initialize vector database

    Args:
        clear_existing: Whether to clear existing collection
        file_path: Specific file path to index
        directory: Directory path to index
        delete_by_filename: Delete documents by filename
    """
    print("Initializing vector database...")

    # Initialize retriever
    retriever = ChromaRetriever()

    try:
        # Check vector store status
        vector_store_exists = retriever.check_vector_store_exists()
        print(f"Vector store status: {'Exists' if vector_store_exists else 'Does not exist'}")

        # Clear existing collection (if needed)
        if clear_existing:
            print("Clearing existing collection...")
            retriever.clear_collection()
            print("Collection successfully cleared")

        # Delete documents by filename (if needed)
        if delete_by_filename:
            print(f"Deleting all documents with filename: {delete_by_filename}...")
            deleted_count = retriever.delete_documents_by_filename(delete_by_filename)
            print(f"Deleted {deleted_count} documents")

        # Check current document count
        current_count = retriever.get_collection_count()
        print(f"Current document count in collection: {current_count}")

        # Index specific file
        if file_path:
            if Path(file_path).exists():
                print(f"Indexing file: {file_path}")
                chunks_count = retriever.index_file(
                    file_path, delete_existing=False
                )  # Don't delete duplicates
                print(f"Successfully indexed {chunks_count} chunks from {file_path}")
            else:
                print(f"Error: File {file_path} does not exist")
                return False

        # Index all files in directory
        elif directory:
            if Path(directory).exists():
                print(f"Indexing directory: {directory}")
                chunks_count = retriever.index_markdown_files(directory, delete_existing=True)
                print(f"Successfully indexed {chunks_count} chunks from {directory}")
            else:
                print(f"Error: Directory {directory} does not exist")
                return False

        # If no file or directory specified, index default data directory
        else:
            settings = get_settings()
            default_data_dir = settings.data_dir
            if Path(default_data_dir).exists():
                print(f"Indexing default data directory: {default_data_dir}")
                chunks_count = retriever.index_markdown_files(
                    default_data_dir, delete_existing=True
                )
                print(f"Successfully indexed {chunks_count} chunks from {default_data_dir}")
            else:
                print(f"Warning: Default data directory {default_data_dir} does not exist")
                print("Please use --file or --directory parameter to specify files or directories to index")

        # Show final document count
        final_count = retriever.get_collection_count()
        print(f"Vector database initialization completed! Final document count: {final_count}")

        return True

    except (ValueError, RuntimeError, FileNotFoundError) as e:
        print(f"Error initializing vector database: {e}")
        return False


def show_collection_info():
    """Show collection information"""
    print("Getting vector database information...")

    retriever = ChromaRetriever()

    try:
        count = retriever.get_collection_count()
        print(f"Current document count in collection: {count}")

        settings = get_settings()
        print(f"Collection name: {settings.collection_name}")
        print(f"Vector database path: {settings.index_dir}")
        print(f"Embedding model: {settings.ollama_embed}")
        print(f"Data directory: {settings.data_dir}")

        return True

    except (ValueError, RuntimeError) as e:
        print(f"Error getting collection information: {e}")
        return False


def test_retrieval(query="What is ECU system?", k=3):
    """Test retrieval function"""
    print(f"Testing retrieval function, query: '{query}'")

    retriever = ChromaRetriever()

    try:
        results = retriever.retrieve(query, k=k)

        print(f"Retrieved {len(results)} relevant documents:")
        for i, doc in enumerate(results, 1):
            print(f"\n--- Document {i} ---")
            print(f"Source: {doc['metadata'].get('filename', 'Unknown')}")
            print(f"Similarity score: {doc['score']:.4f}")
            print(f"Content preview: {doc['content'][:200]}...")

        return True

    except (ValueError, RuntimeError) as e:
        print(f"Error testing retrieval function: {e}")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Initialize and manage vector database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Initialize vector database (using default data directory)
  python init_vector_db.py
  
  # Clear existing data and reinitialize
  python init_vector_db.py --clear
  
  # Index specific file
  python init_vector_db.py --file data/ECU-700_Series_Manual.md
  
  # Index specific directory
  python init_vector_db.py --directory ./my_documents
  
  # Show collection information only
  python init_vector_db.py --info
  
  # Test retrieval function
  python init_vector_db.py --test
  
  # Test specific query
  python init_vector_db.py --test --query "Main functions of ECU system"
        """,
    )

    parser.add_argument("--clear", action="store_true", help="Clear existing collection before initialization")
    parser.add_argument("--file", type=str, help="Index specific file")
    parser.add_argument("--directory", type=str, help="Index all files in specific directory")
    parser.add_argument("--delete-by-filename", type=str, help="Delete documents by filename")
    parser.add_argument("--info", action="store_true", help="Show collection information")
    parser.add_argument("--test", action="store_true", help="Test retrieval function")
    parser.add_argument("--query", type=str, default="What is ECU system?", help="Query text for testing retrieval")
    parser.add_argument("--k", type=int, default=3, help="Number of documents to return during retrieval")

    args = parser.parse_args()

    # Show collection information
    if args.info:
        success = show_collection_info()
        sys.exit(0 if success else 1)

    # Test retrieval function
    if args.test:
        success = test_retrieval(args.query, args.k)
        sys.exit(0 if success else 1)

    # Initialize vector database
    success = initialize_vector_db(
        clear_existing=args.clear,
        file_path=args.file,
        directory=args.directory,
        delete_by_filename=args.delete_by_filename,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
