from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "model.pkl"
SCALER_PATH = BASE_DIR / "Scaler.pkl"


# --------------------------------------------------
# 2. Features in EXACT training order
# --------------------------------------------------

FEATURES = [
    "Vibration_Amplitude",
    "RMS_Vibration",
    "Vibration_Frequency",
    "Surface_Temperature",
    "Exhaust_Temperature",
    "Acoustic_dB",
    "Acoustic_Frequency",
    "Intake_Pressure",
    "Exhaust_Pressure",
    "Frequency_Band_Energy",
    "Amplitude_Mean",
]


# --------------------------------------------------
# 3. Load model and fitted scaler
# --------------------------------------------------

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

app = FastAPI(title="Engine Condition Predictor")

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)


# --------------------------------------------------
# 4. Input validation
# --------------------------------------------------

class SensorInput(BaseModel):
    Vibration_Amplitude: float = Field(ge=0)
    RMS_Vibration: float = Field(ge=0)
    Vibration_Frequency: float = Field(ge=0)

    Surface_Temperature: float
    Exhaust_Temperature: float
    Acoustic_dB: float

    Acoustic_Frequency: float = Field(ge=0)
    Intake_Pressure: float
    Exhaust_Pressure: float

    Frequency_Band_Energy: float = Field(ge=0)
    Amplitude_Mean: float


# --------------------------------------------------
# 5. Website homepage
# --------------------------------------------------

@app.get("/")
def home():
    return FileResponse(
        BASE_DIR / "templates" / "index.html"
    )


# --------------------------------------------------
# 6. Health check
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
    }


# --------------------------------------------------
# 7. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: SensorInput):

    # Convert inputs to a DataFrame with exact column names.
    input_data = pd.DataFrame(
        [[getattr(data, feature) for feature in FEATURES]],
        columns=FEATURES,
    )

    # Step 1: Apply the SAME scaler fitted during training.
    X_scaled = scaler.transform(input_data)

    # Step 2: Predict using the trained Random Forest.
    prediction = model.predict(X_scaled)[0]

    result = {
        "prediction": str(prediction),
        "condition": str(prediction),
    }

    # Step 3: Return probabilities if supported.
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_scaled)[0]

        result["probabilities"] = {
            str(label): round(float(probability), 4)
            for label, probability in zip(
                model.classes_,
                probabilities,
            )
        }

    return result