FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "ai_trading_engine.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
