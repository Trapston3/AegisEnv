FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir . uvicorn[standard]
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
