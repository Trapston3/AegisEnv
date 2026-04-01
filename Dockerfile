FROM python:3.11-slim

# Install uv via official installer
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the environment structure
COPY . .

# Install dependencies using uv sync
RUN uv sync --frozen

# Expose port and run the server explicitly over 7860 for HF Spaces
EXPOSE 7860
CMD ["uv", "run", "python", "-m", "server.app", "--host", "0.0.0.0", "--port", "7860"]
