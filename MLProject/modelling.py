import argparse, pandas as pd, mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def main(data_path: str):
    df = pd.read_csv(data_path)
    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mlflow.log_metrics({"accuracy": accuracy_score(y_test, y_pred), "f1_score": f1_score(y_test, y_pred)})
        mlflow.sklearn.log_model(model, "trained_model")
        print("✅ Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing/diabetes_preprocessed.csv")
    args = parser.parse_args()
    main(args.data_path)
