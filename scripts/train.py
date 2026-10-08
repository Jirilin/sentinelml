from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import platform
import joblib
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from app.core.config import MODEL_PATH, METADATA_PATH, MODEL_VERSION, REFERENCE_PATH
from app.ml.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

DATA = ROOT / "data" / "raw" / "service_requests.csv"

def main():
    if not DATA.exists(): raise SystemExit("Dataset missing. Run: python scripts/generate_data.py")
    df = pd.read_csv(DATA)
    X, y = df[ALL_FEATURES], df["sla_breached"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer([("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)])
    model = Pipeline([("preprocess", pre), ("classifier", HistGradientBoostingClassifier(max_iter=180, learning_rate=.08, max_leaf_nodes=20, class_weight="balanced", random_state=42))])
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    pred = (proba >= .5).astype(int)
    metrics = {"roc_auc": roc_auc_score(y_test, proba), "f1": f1_score(y_test, pred), "accuracy": accuracy_score(y_test, pred)}
    run_id = "mlflow-not-installed"
    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
        mlflow.set_experiment("sentinelml-service-risk")
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            mlflow.log_params({"algorithm":"HistGradientBoostingClassifier", "max_iter":180, "learning_rate":.08, "max_leaf_nodes":20, "class_weight":"balanced"})
            mlflow.log_metrics(metrics)
            signature = mlflow.models.infer_signature(X_train.head(10), model.predict_proba(X_train.head(10)))
            mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train.head(3))
    else:
        print("MLflow is not installed; training continues without experiment tracking.")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    metadata = {"model_version": MODEL_VERSION, "run_id": run_id, "metrics": metrics, "python": platform.python_version(), "scikit_learn": sklearn.__version__, "training_rows": len(X_train), "features": ALL_FEATURES}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train.sample(min(2000, len(X_train)), random_state=42).to_csv(REFERENCE_PATH, index=False)
    print(json.dumps(metadata, indent=2))
if __name__ == "__main__": main()
