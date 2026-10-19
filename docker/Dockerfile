FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config/gateway.example.yaml config/gateway.example.yaml
RUN pip install --no-cache-dir ".[server]"
EXPOSE 8080
ENTRYPOINT ["uvicorn", "llm_gateway.server:app", "--host", "0.0.0.0", "--port", "8080"]
