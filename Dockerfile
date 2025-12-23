# Dockerfile for Hugging Face Spaces (Docker runtime) or generic container deploys
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV PORT=7860
EXPOSE 7860
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]

