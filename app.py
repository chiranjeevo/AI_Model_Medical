from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd
from rapidfuzz import process
import uuid

# ----------------------------------
# Load Model & Encoders
# ----------------------------------
model = joblib.load("disease_model.pkl")
mlb = joblib.load("mlb.pkl")
enc = joblib.load("encoder.pkl")
known_symptoms = list(mlb.classes_)

# ----------------------------------
# App
# ----------------------------------
app = FastAPI(title="Conversational Disease Prediction API")

# ----------------------------------
# CORS (VERY IMPORTANT)
# ----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (safe for demo)
    allow_credentials=True,
    allow_methods=["*"],   # allow POST, GET, OPTIONS
    allow_headers=["*"],
)

# ----------------------------------
# In-Memory Session Storage
# ----------------------------------
sessions = {}

# ----------------------------------
# Utility Functions
# ----------------------------------
def map_symptom(symptom, known_list, threshold=80):
    match, score, _ = process.extractOne(symptom.lower(), known_list)
    return match if score >= threshold else None

def categorize_days(days):
    if days <= 3:
        return "acute"
    elif days <= 14:
        return "sub-acute"
    return "chronic"

# ----------------------------------
# Request Models
# ----------------------------------
class StartRequest(BaseModel):
    symptoms: List[str]
    days: int
    severity: int
    age_group: str
    gender: str

class AnswerRequest(BaseModel):
    session_id: str
    yes_symptoms: List[str]
    top_k: Optional[int] = 3

# ----------------------------------
# STEP 1 – Start Diagnosis
# ----------------------------------
@app.post("/start")
def start_diagnosis(data: StartRequest):

    mapped = []
    for s in data.symptoms:
        m = map_symptom(s, known_symptoms)
        if m:
            mapped.append(m)

    if not mapped:
        return {"error": "No matching symptoms found"}

    remaining = list(set(known_symptoms) - set(mapped))

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "symptoms": mapped,
        "days": data.days,
        "severity": data.severity,
        "age_group": data.age_group,
        "gender": data.gender
    }

    return {
        "session_id": session_id,
        "questions": remaining
    }

# ----------------------------------
# STEP 2 – Final Prediction
# ----------------------------------
@app.post("/final")
def final_prediction(data: AnswerRequest):

    if data.session_id not in sessions:
        return {"error": "Invalid session ID"}

    session = sessions[data.session_id]
    final_symptoms = list(set(session["symptoms"] + data.yes_symptoms))

    symptom_vector = mlb.transform([final_symptoms])
    df = pd.DataFrame(symptom_vector, columns=mlb.classes_)

    df["Days"] = session["days"]
    df["Severity_Avg"] = session["severity"]
    df["Symptom_Count"] = len(final_symptoms)

    cat = enc.transform([[
        categorize_days(session["days"]),
        session["age_group"],
        session["gender"]
    ]])

    cat_df = pd.DataFrame(
        cat,
        columns=enc.get_feature_names_out(
            ["Duration_Category", "Age_Group", "Gender"]
        )
    )

    df = pd.concat([df, cat_df], axis=1)
    df = df.reindex(columns=model.feature_names_in_, fill_value=0)

    probs = model.predict_proba(df)[0]

    result = pd.DataFrame({
        "Disease": model.classes_,
        "Probability": probs
    }).sort_values(by="Probability", ascending=False).head(data.top_k)

    del sessions[data.session_id]

    return {
        "status": "success",
        "predictions": result.to_dict(orient="records")
    }

# ----------------------------------
# Root Endpoint
# ----------------------------------
@app.get("/")
def root():
    return {"message": "Disease Prediction API is running 🚀"}
