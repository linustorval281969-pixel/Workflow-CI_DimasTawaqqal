FROM python:3.10-slim

WORKDIR /app


COPY . /app/


RUN pip install --no-cache-dir pandas numpy scikit-learn mlflow dagshub


CMD ["python", "modelling.py", "--data_path", "diabetes_preprocessing/diabetes_preprocessed.csv"]
