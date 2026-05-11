import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main(data_path: str):
    # Load Data
    df = pd.read_csv(data_path)
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # ✅ FIX: JANGAN pakai set_experiment() saat run via Docker/mlflow run
    # Biarkan MLflow pakai default experiment agar tidak bentrok
    
    # ✅ FIX: JANGAN pakai run_name, biarkan MLflow handle otomatis
    with mlflow.start_run():
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Manual Logging (Advanced Requirement)
        mlflow.log_param("algorithm", "RandomForestClassifier")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metrics({"accuracy": acc, "f1_score": f1})
        
        # Log model artifact
        mlflow.sklearn.log_model(model, "trained_model")
        
        print(f"✅ MLflow Project selesai. Accuracy: {acc:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing/diabetes_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)