from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Loan Approval ML API")

# Load both trained models
model_rf = joblib.load("ml_model_api/model_rf.pkl")
model_lr = joblib.load("ml_model_api/model_lr.pkl")


class LoanInput(BaseModel):
    model: str

    no_of_dependents: int
    education: str
    self_employed: str
    loan_amount: float
    loan_term: int
    cibil_score: float
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float
    income_annum: float


@app.get("/")
def home():
    return {"message": "Loan Approval API is running"}


@app.post("/predict")
def predict(data: LoanInput):

    # Choose the model
    if data.model.lower() == "rf":
        model = model_rf
        model_name = "Random Forest"

    elif data.model.lower() == "lr":
        model = model_lr
        model_name = "Logistic Regression"

    else:
        return {
            "error": "Invalid model. Use 'rf' or 'lr'."
        }

    # Calculate loan_to_income
    loan_to_income = data.loan_amount / data.income_annum

    # Prepare input
    input_data = pd.DataFrame([{
        "no_of_dependents": data.no_of_dependents,
        "education": data.education,
        "self_employed": data.self_employed,
        "loan_term": data.loan_term,
        "cibil_score": data.cibil_score,
        "residential_assets_value": data.residential_assets_value,
        "commercial_assets_value": data.commercial_assets_value,
        "luxury_assets_value": data.luxury_assets_value,
        "bank_asset_value": data.bank_asset_value,
        "loan_to_income": loan_to_income
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    # Probability
    probabilities = model.predict_proba(input_data)[0]
    probability = float(max(probabilities))

    return {
        "model": model_name,
        "prediction": str(prediction),
        "probability": probability
    }