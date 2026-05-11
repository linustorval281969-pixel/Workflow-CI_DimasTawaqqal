FROM python:3.10-slim

WORKDIR /app

# Copy seluruh isi MLProject ke dalam container
COPY . /app/

# Install dependencies
RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow dagshub

# Default command: jalankan MLflow Project entry point
CMD ["python", "modelling.py", "--data_path", "diabetes_preprocessing/diabetes_preprocessed.csv"]