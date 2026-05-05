FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY dashboard /app/dashboard
COPY scripts /app/scripts
COPY .env.example /app/.env.example

EXPOSE 8000

CMD ["uvicorn", "anpr.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
