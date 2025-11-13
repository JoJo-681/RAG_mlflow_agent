# Use Python 3.11 slim image
FROM python:3.11-slim-bookworm

WORKDIR /app
# ENV HTTP_PROXY=http://localhost:15236
# ENV HTTPS_PROXY=http://localhost:15236

# Install system dependencies using Aliyun mirror (for CN)
RUN rm -f /etc/apt/sources.list.d/* && \
    echo "deb http://mirrors.aliyun.com/debian bookworm main contrib non-free non-free-firmware\n\
deb http://mirrors.aliyun.com/debian bookworm-updates main contrib non-free non-free-firmware\n\
deb http://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
    apt-get clean && apt-get update && apt-get install -y \
        curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

# Copy poetry files first (for dependency caching)
COPY pyproject.toml README.md ./

# Install dependencies (skip installing current project to avoid README.md/package errors)
RUN poetry install --only main --no-root --no-interaction --no-ansi

# Copy rest of project files
COPY src/ ./src/
COPY data/ ./data/
COPY init_vector_db.py mlflow_run.py mlflow_start.py mlflow_deploy.py ./ 
COPY test_questions_script.py test_single_query.py ./ 
COPY test-questions.csv questions.json ./
COPY .env.example ./
COPY .env ./

# Create directories
RUN mkdir -p artifacts/chroma_index

EXPOSE 5000
EXPOSE 5001

# CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlruns"]
CMD [ "python", "mlflow_run.py" ]