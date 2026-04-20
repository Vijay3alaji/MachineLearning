# 1. Base image — use the same Python version as your venv
FROM python:3.10-slim


WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "application.py"]