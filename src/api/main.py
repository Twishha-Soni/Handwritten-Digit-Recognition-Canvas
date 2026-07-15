from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import joblib
import numpy as np
from pydantic import BaseModel, field_validator
from scipy.ndimage import gaussian_filter

app = FastAPI()

# --- load the trained pipeline ONCE at startup, not per-request ---
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / 'models' / 'digit_pipeline.pkl'

if not model_path.exists():
    raise RuntimeError(
        f"No trained model found at {model_path}."
        f"Run 'python -m src.training.train' first."
    )

pipeline = joblib.load(model_path)

# --- schemas: the explicit contract for req/res shape
class DigitRequest(BaseModel):
    pixels: list[float]

    @field_validator('pixels')
    @classmethod
    def check_length(cls, value: list[float]) -> list[float]:
        if len(value) != 784:
            raise ValueError(f"Expected 784 pixels values, got {len(value)}")
        return value
    
    @field_validator('pixels')
    @classmethod
    def check_range(cls, value: list[float]) -> list[float]:
        arr = np.array(value, dtype=np.float64)
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise ValueError('Pixel values must be finite numbers (No NaN/Inf)')
        if arr.min() < 0 or arr.max() > 255:
            raise ValueError('Pixel values must be in range 0-255')
        return value



class DigitResponse(BaseModel):
    predicted_digit: str
    probabilities: dict[str, float]

def preprocess(raw_pixels: list[float]) -> np.ndarray:
    """Transform raw canvas pixels into MNIST-like pixels: inverted, centered."""
    image = np.array(raw_pixels, dtype=np.float64).reshape(28,28)

    # 1. invert: canvas is black-ink-on-white, MNIST is white-ink-on-black
    image = 255.0 - image

    # 2. center the digit within the 28x28 frame using its bounding box
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)

    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0,-1]]
        col_min, col_max = np.where(cols)[0][[0,-1]]

        digit = image[row_min:row_max + 1, col_min:col_max + 1]

        centered = np.zeros((28,28), dtype=np.float64)
        h, w = digit.shape
        top = (28 - h) // 2
        left = (28  - w) // 2
        centered[top:top + h, left:left + w] = digit
        image = centered
    
    image = gaussian_filter(image, sigma=1.0)

    return image.flatten()

@app.get('/')
def serve_frontend():
    frontend_path = project_root / 'src' / 'frontend' / 'index.html'
    return FileResponse(frontend_path)

@app.post('/predict', response_model=DigitResponse)
def predict(request: DigitRequest):
    processed_pixels = preprocess(request.pixels)
   
    if all(processed_pixels == 0):
        raise HTTPException(status_code=400, detail='Canvas appears blank - draw a digit first.')
    
    proba = pipeline.predict_proba([processed_pixels])[0]
    classes = pipeline.classes_
    probabilities = {str(cls): float(p) for cls,p in zip(classes, proba)}

    prediction = pipeline.predict([processed_pixels])
    print(prediction[0])
    return DigitResponse(predicted_digit=prediction[0], probabilities=probabilities)

