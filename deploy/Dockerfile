FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the package
RUN pip install --no-cache-dir .

# Expose port
EXPOSE 8000

# Default command (can be overridden to run edge_api)
CMD ["uvicorn", "m2m.api.coordinator_api:app", "--host", "0.0.0.0", "--port", "8000"]
