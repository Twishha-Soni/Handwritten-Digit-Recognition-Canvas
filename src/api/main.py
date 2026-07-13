from fastapi import FastAPI
from fastapi.responses import FileResponse
from pathlib import Path
import joblib
from pydantic import BaseModel, field_validator

app = FastAPI()

# --- load the trained pipeline ONCE at startup, not per-request ---
project_root = Path(__file__).resolve().parents[2]
model_path = project_root / 'models' / 'digit_pipeline.pkl'
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


class DigitResponse(BaseModel):
    predicted_digit: str

@app.get('/')
def serve_frontend():
    frontend_path = project_root / 'src' / 'frontend' / 'index.html'
    return FileResponse(frontend_path)

@app.post('/predict', response_model=DigitResponse)
def predict(request: DigitRequest):
    prediction = pipeline.predict([request.pixels])
    return DigitResponse(predicted_digit=prediction[0])

