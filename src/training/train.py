# train.py
import joblib
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# --- load data ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist["data"], mnist["target"]

# --- split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- build pipeline: scaler + classifier as ONE object ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
])

# --- train ---
pipeline.fit(X_train, y_train)

# --- evaluate ---
accuracy = pipeline.score(X_test, y_test)
print("Test accuracy:", accuracy)

# --- persist the ENTIRE fitted pipeline to disk ---
project_root = Path(__file__).resolve().parents[2]  # src/training/train.py -> project root
models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)

output_path = models_dir / "digit_pipeline.joblib"
joblib.dump(pipeline, output_path)
print(f"Pipeline saved to {output_path}")