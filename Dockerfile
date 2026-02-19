# Dockerfile for Aflac Assist Chat
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and ChromaDB
RUN mkdir -p /app/data /app/chroma

# Expose ports
EXPOSE 8000 11434

# Create startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
