# train.py
import joblib
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

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
    ("clf", SGDClassifier(random_state=42)),
])

# --- train ---
pipeline.fit(X_train, y_train)

# --- evaluate ---
accuracy = pipeline.score(X_test, y_test)
print("Test accuracy:", accuracy)

# --- persist the ENTIRE fitted pipeline to disk ---
joblib.dump(pipeline, "digit_pipeline.pkl")
print("Pipeline saved to digit_pipeline.pkl")