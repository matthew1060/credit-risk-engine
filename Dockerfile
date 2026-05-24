FROM python:3.10-slim

WORKDIR /app

COPY requirements_hf.txt .
RUN pip install --no-cache-dir -r requirements_hf.txt

COPY models/ ./models/
COPY src/ ./src/

EXPOSE 8501

CMD ["streamlit", "run", "src/dashboard/app_standalone.py", "--server.port=8501", "--server.address=0.0.0.0"]