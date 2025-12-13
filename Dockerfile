FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Use a non-root user if available (keeps container safer)
RUN adduser --disabled-password --gecos "" appuser || true
USER appuser

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
