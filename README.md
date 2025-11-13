# RAG System with MLflow and LangChain

A production-ready Retrieval-Augmented Generation (RAG) system for document-based question answering, built with modern AI/ML tools including MLflow, LangChain, ChromaDB, and DeepSeek LLM.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- DeepSeek API Key
- Ollama (for local embeddings)

### Installation & Setup

1. **Clone and setup the project**
```bash
cd test_project
```

2. **Install dependencies**
```bash
poetry install
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys
```

4. **Initialize vector database**
```bash
python init_vector_db.py
```

5. **Start MLflow tracking server**
```bash
python mlflow_start.py
```

6. **Train and log the model**
```bash
python mlflow_run.py
```

7. **Deploy the model service**
```bash
python mlflow_deploy.py
```

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Overview

This project implements a comprehensive RAG system that combines document retrieval with large language models to provide accurate, context-aware answers to user questions. The system is designed for production use with MLflow for model tracking, serving, and management.

### Key Components
- **Document Processing**: Automatic chunking and embedding of documents
- **Vector Search**: ChromaDB for efficient similarity search
- **LLM Integration**: DeepSeek API and Ollama local models
- **Model Management**: MLflow for experiment tracking and model serving
- **Production Deployment**: Docker containerization and REST API

## Architecture

### System Flow
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│   User Query    │ →  │   RAG Agent     │ →  │   Vector Store  │ →  │   LLM       │
│                 │    │                 │    │   (ChromaDB)    │    │   (DeepSeek)│
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
         ↓                      ↓                      ↓                    ↓
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────┐
│   Response      │ ←  │   Context       │ ←  │   Relevant      │ ←  │   LLM       │
│                 │    │   Builder       │    │   Documents     │    │   Output    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────┘
```

### MLflow Integration
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Model         │ →  │   MLflow        │ →  │   MLflow        │
│   Training      │    │   Tracking      │    │   Serving       │
│                 │    │   Server        │    │   API           │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

### Core RAG Features
- **Intelligent Document Retrieval**: Vector-based semantic search using ChromaDB
- **Multi-format Support**: Process PDF, DOCX, TXT, and Markdown files
- **Context-Aware Generation**: LLM responses based on retrieved document context
- **Batch Processing**: Handle multiple queries efficiently

### ML & MLOps Features
- **Experiment Tracking**: Comprehensive logging with MLflow
- **Model Versioning**: Track model iterations and performance
- **Model Serving**: REST API for production deployment
- **Performance Monitoring**: Track query metrics and response quality

### Production Features
- **Docker Containerization**: Easy deployment and scaling
- **Configuration Management**: Environment-based settings
- **Error Handling**: Robust error recovery and logging
- **Vector Store Management**: Automatic validation and recovery

## Project Structure

```
.
├── src/                       # Source code
│   ├── rag/                   # RAG implementation
│   │   ├── __init__.py
│   │   ├── graph.py           # LangGraph workflow
│   │   ├── retriever.py       # ChromaDB retriever
│   │   └── chain.py           # LangChain RAG chain
│   ├── models/                # Data models
│   │   ├── __init__.py
│   │   ├── schemas.py         # Pydantic schemas
│   │   └── mlflow_rag_model.py # MLflow model wrapper
│   ├── llm/                   # LLM clients
│   │   ├── __init__.py
│   │   └── openai_client.py   # DeepSeek client
│   ├── utils/                 # Utilities
│   │   ├── __init__.py
│   │   └── mlflow_tracker.py  # MLflow tracking utilities
│   └── config.py              # Configuration management
├── data/                      # Document storage
│   ├── ECU-700_Series_Manual.md
│   ├── ECU-800_Series_Base.md
│   └── ECU-800_Series_Plus.md
├── docker/                    # Docker configuration
│   └── Dockerfile
├── artifacts/                 # Vector database and model artifacts
│   └── chroma_index/          # ChromaDB vector store
├── mlruns/                    # MLflow experiment tracking
├── mlflow.db/                 # MLflow database
├── docker-compose.yml         # Docker Compose configuration
├── pyproject.toml            # Poetry dependencies and project config
├── .env.example              # Environment variables template
├── .env                      # Environment variables (user created)
├── .dockerignore             # Docker ignore patterns
├── init_vector_db.py         # Initialize vector database
├── mlflow_start.py           # Start MLflow tracking server
├── mlflow_run.py             # Train and log model to MLflow
├── mlflow_deploy.py          # Deploy model service
├── test_single_query.py      # Single query testing
├── test_questions_script.py  # Batch testing script
├── test_mlflow_integration.py # MLflow integration testing
├── check_mlflow_version.py   # MLflow version compatibility check
├── questions.json            # Test questions for batch testing
├── test-questions.csv        # CSV test questions
├── test_results.csv          # Test results output
└── README.md                 # Project documentation


```

## Installation

### Method 1: Local Development

1. **Install Python dependencies**
```bash
poetry install
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

3. **Start required services**
```bash
# Start Ollama for embeddings
ollama serve

# Start MLflow tracking server
python mlflow_start.py
```

### Method 2: Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

## Usage

### Vector Database Management

**Initialize vector store:**
```bash
python init_vector_db.py
```

**Manage vector store:**
```bash
# Check vector store status
python init_vector_db.py --info

# Clear and recreate vector store
python init_vector_db.py --clear

# Add specific document
python init_vector_db.py --file data/new_document.md

# Test retrieval
python init_vector_db.py --test --query "your question"
```

## MLflow Workflow

The MLflow workflow can be executed using either Python scripts or manual CLI commands. Below is the complete sequential workflow:

### Step 1: Start MLflow Tracking Server

**Using Python Script:**
```bash
python mlflow_start.py
```

**Using Manual CLI Commands:**
```bash
# Basic MLflow server (Linux/Mac)
mlflow server --host localhost --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns

# Windows compatible command
mlflow server --host localhost --port 5000 --backend-store-uri ./mlruns --default-artifact-root ./mlruns --workers 1
```

**Stop MLflow server:**
- Press `Ctrl+C` in the terminal where MLflow server is running
- Or kill the process using task manager/process manager

### Step 2: Train and Log Model

**Using Python Script:**
```bash
python mlflow_run.py
```

**Using Manual CLI Commands:**
```bash
# Run the training script (this logs model to MLflow)
python mlflow_run.py

# Save the run_id from output for deployment
```

### Step 3: Deploy Model Service

**Using Python Script:**
```bash
python mlflow_deploy.py
```

**Using Manual CLI Commands:**
```bash
# Deploy model using run_id from training
mlflow models serve --model-uri runs:/<run_id>/rag_agent --port 5001 --host 0.0.0.0 --env-manager local

# Example with actual run_id
mlflow models serve --model-uri runs:/0c32e56289204ef2961c3efac471c8f7/rag_agent --port 5001 --host 0.0.0.0 --env-manager local

# For Windows compatibility, use localhost instead of 0.0.0.0
mlflow models serve --model-uri runs:/<run_id>/rag_agent --port 5001 --host localhost --env-manager local
```

**Note:** Replace `<run_id>` with the actual run ID from your model training output.

## API Documentation

### Query Endpoint

**URL**: `http://localhost:5001/invocations`

**Method**: `POST`

**Content-Type**: `application/json`

### Request Format

**Single Query:**
```json
{
  "dataframe_split": {
    "columns": ["query"],
    "data": [["What is the maximum operating temperature for the ECU-750?"]]
  }
}
```

**Batch Queries:**
```json
{
  "dataframe_split": {
    "columns": ["query"],
    "data": [
      ["Question 1"],
      ["Question 2"],
      ["Question 3"]
    ]
  }
}
```

### Response Format

```json
{
  "predictions": [
    {
      "answer": "The maximum operating temperature for the ECU-750 is 85°C...",
      "source_documents": 3,
      "status": "success"
    }
  ]
}
```

### Testing Tools

**Single query testing:**
```bash
python test_single_query.py "Your question here"
```

**Batch testing:**
```bash
python test_questions_script.py
```

**MLflow integration testing:**
```bash
python test_mlflow_integration.py
```

**Using curl (Linux/Mac):**
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["query"], "data": [["Your question"]]}}'
```

**Using curl (Windows Command Prompt):**
```cmd
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_split\": {\"columns\": [\"query\"], \"data\": [[\"Your question\"]]}}"
```

**Using curl with questions.json file (Linux/Mac):**
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d @questions.json
```

**Using curl with questions.json file (Windows Command Prompt):**
```cmd
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d "@questions.json"
```

## Multiple Question Testing

The system supports several methods for testing multiple questions, allowing you to evaluate performance across different queries.

### Method 1: Using questions.json File

The project includes a `questions.json` file that contains predefined test questions. This file follows the MLflow model serving format:

**questions.json content:**
```json
{
  "dataframe_split": {
    "columns": ["query"],
    "data": [
      ["What is the maximum operating temperature for the ECU-750?"],
      ["How do you enable the NPU on the ECU-850b?"],
      ["Which ECU can operate in the harshest temperature conditions?"]
    ]
  }
}
```

**Usage:**
```bash
# Using curl with questions.json
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d @questions.json
```

### Method 2: Custom Batch Testing Script

Use the built-in testing script for comprehensive batch testing:

```bash
python test_questions_script.py
```

This script will:
- Load test questions from `test-questions.csv`
- Send each question to the RAG system
- Compare responses with reference answers
- Generate detailed performance metrics
- Save results to `test_results.csv`

### Method 3: Direct API Calls with Multiple Questions

You can send multiple questions directly in a single API call:

**Linux/Mac:**
```bash
curl -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["query"], "data": [["Question 1"], ["Question 2"], ["Question 3"]]}}'
```

**Windows Command Prompt:**
```cmd
curl -X POST http://localhost:5001/invocations -H "Content-Type: application/json" -d "{\"dataframe_split\": {\"columns\": [\"query\"], \"data\": [[\"Question 1\"], [\"Question 2\"], [\"Question 3\"]]}}"
```

### Method 4: Interactive Testing

For interactive testing of multiple questions:

```bash
python test_single_query.py
```

This will start an interactive session where you can:
- Enter questions one by one
- See immediate responses
- Test different question types
- Exit when finished

### Creating Custom Test Sets

To create your own test questions:

**Modify questions.json:**
   ```json
   {
     "dataframe_split": {
       "columns": ["query"],
       "data": [
         ["Your question 1"],
         ["Your question 2"],
         ["Your question 3"]
       ]
     }
   }
   ```

### Expected Response Format for Multiple Questions

When testing multiple questions, the API returns responses for all questions:

```json
{
  "predictions": [
    {
      "answer": "Answer to question 1...",
      "source_documents": 3,
      "status": "success"
    },
    {
      "answer": "Answer to question 2...", 
      "source_documents": 2,
      "status": "success"
    },
    {
      "answer": "Answer to question 3...",
      "source_documents": 4,
      "status": "success"
    }
  ]
}
```

## Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# DeepSeek API Configuration
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_MODEL=deepseek-chat

# Ollama Embeddings
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_EMBED=qllama/bge-large-en-v1.5:latest

# Vector Database
INDEX_DIR=artifacts/chroma_index
COLLECTION_NAME=rag_documents
DATA_DIR=data

# Document Processing
CHUNK_SIZE=1200
CHUNK_OVERLAP=150
TOP_K=8

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=rag_experiment
```

### Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `DEEPSEEK_API_KEY` | DeepSeek API key | Required |
| `OLLAMA_EMBED` | Embedding model name | `qllama/bge-large-en-v1.5:latest` |
| `CHUNK_SIZE` | Document chunk size | `1200` |
| `CHUNK_OVERLAP` | Chunk overlap | `150` |
| `TOP_K` | Documents to retrieve | `8` |
| `INDEX_DIR` | Vector store location | `artifacts/chroma_index` |

## Docker Deployment

### Using Docker Compose

```bash
# Start all services
docker-compose up --build

# Stop services
docker-compose down

# View logs
docker-compose logs -f
```

### Manual Docker Deployment

```bash
# Build image
docker build -t rag-mlflow -f docker/Dockerfile .

# Run container
docker run -d \
  --name rag-mlflow \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/mlruns:/app/mlruns \
  -e DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY} \
  rag-mlflow
```

### Services Available

- **MLflow UI**: http://localhost:5000
- **Model API**: http://localhost:5001/invocations

## Development

### Adding Documents

Place documents in the `data/` directory. Supported formats:
- Markdown (.md)
- Text (.txt)
- PDF (.pdf)
- Word documents (.docx)

### Testing

**Run test suite:**
```bash
poetry run pytest

# With coverage
poetry run pytest --cov=src tests/
```

**Test specific components:**
```bash
# Test vector store
python init_vector_db.py --test

# Test single query
python test_single_query.py

# Batch testing
python test_questions_script.py

# MLflow integration testing
python test_mlflow_integration.py
```

### Code Structure

- `src/rag/`: Core RAG implementation
- `src/models/`: Data models and MLflow integration
- `src/llm/`: LLM client implementations
- `src/utils/`: Utility functions and tracking

## Troubleshooting

### Common Issues

**Vector store initialization fails:**
```bash
# Check if data directory exists
ls data/

# Reinitialize vector store
python init_vector_db.py --clear

# Check if embedding model is available
curl http://127.0.0.1:11434/api/tags

# Pull embedding model if missing
ollama pull qllama/bge-large-en-v1.5
```

**Ollama service not available:**
```bash
# Start Ollama service
ollama serve
