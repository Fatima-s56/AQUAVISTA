
import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="AQUAVISTA Water Quality API")

model = joblib.load("aquavista_decision_tree.pkl")
feature_names = joblib.load("aquavista_feature_names.pkl")


class WaterQualityInput(BaseModel):
    aluminium: float
    ammonia: float
    arsenic: float
    barium: float
    cadmium: float
    chloramine: float
    chromium: float
    copper: float
    flouride: float
    bacteria: float
    viruses: float
    lead: float
    nitrates: float
    nitrites: float
    mercury: float
    perchlorate: float
    radium: float
    selenium: float
    silver: float
    uranium: float


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AQUAVISTA</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">

        <style>
            body {
                margin: 0;
                font-family: Arial, sans-serif;
                background: #f2f7f5;
                color: #173d34;
            }

            header {
                background: #073B30;
                color: white;
                text-align: center;
                padding: 55px 20px;
            }

            header h1 {
                font-size: 42px;
                margin-bottom: 10px;
            }

            header p {
                color: #d5e9df;
            }

            .container {
                max-width: 850px;
                margin: 35px auto;
                padding: 20px;
            }

            .card {
                background: white;
                padding: 25px;
                margin-bottom: 20px;
                border-radius: 14px;
                box-shadow: 0 3px 12px #00000012;
            }

            .card h2 {
                color: #075B46;
            }

            .button {
                display: inline-block;
                background: #075B46;
                color: white;
                padding: 12px 22px;
                border-radius: 8px;
                text-decoration: none;
                margin-top: 10px;
            }

            .button:hover {
                background: #073B30;
            }

            footer {
                text-align: center;
                padding: 25px;
                color: #60766e;
            }
        </style>
    </head>

    <body>

        <header>
            <h1>AQUAVISTA</h1>
            <p>An Explainable AI Framework for River Water Quality Assessment and Anomaly Intelligence</p>
            <p>AI-Powered Water Quality Assessment</p>
        </header>

        <div class="container">

            <div class="card">
                <h2>Welcome to AQUAVISTA</h2>
                <p>
                    AQUAVISTA uses Machine Learning to assess water-quality
                    parameters and provide preliminary water safety predictions.
                </p>
            </div>

            <div class="card">
                <h2>1. API Documentation</h2>
                <p>Explore and test the prediction API.</p>
                <a class="button" href="/docs">Open API Documentation</a>
            </div>

            <div class="card">
                <h2>2. ML Prediction API</h2>
                <p>Test the trained Decision Tree model.</p>
                <a class="button" href="/docs#/default/predict_water_predict_post">
                    Test ML Prediction
                </a>
            </div>

            <div class="card">
                <h2>3. API Status</h2>
                <p>Check whether the AQUAVISTA API is running.</p>
                <a class="button" href="/health">Check API Status</a>
            </div>

        </div>

        <footer>
            AQUAVISTA | Machine Learning Project
        </footer>

    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "online",
        "project": "AQUAVISTA"
    }


@app.post("/predict")
def predict_water(data: WaterQualityInput):

    input_data = data.model_dump()

    input_df = pd.DataFrame(
        [input_data],
        columns=feature_names
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    result = "Safe" if prediction == 1 else "Unsafe"

    probability = float(
        probabilities[list(model.classes_).index(prediction)]
    )

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(probability, 4)
    }
