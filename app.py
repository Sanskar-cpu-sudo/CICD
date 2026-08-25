import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field


MODEL_PATH = "model.pkl"


with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


app = FastAPI(
    title="Student Placement Prediction API"
)


class Student(BaseModel):

    cgpa: float = Field(
        ...,
        ge=0,
        le=10
    )

    attendance: float = Field(
        ...,
        ge=0,
        le=100
    )

    coding_score: float = Field(
        ...,
        ge=0,
        le=100
    )

    projects: int = Field(
        ...,
        ge=0
    )

    internships: int = Field(
        ...,
        ge=0
    )

    communication_skills: float = Field(
        ...,
        ge=0,
        le=10
    )


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(student: Student):

    data = [[
        student.cgpa,
        student.attendance,
        student.coding_score,
        student.projects,
        student.internships,
        student.communication_skills,
    ]]

    prediction = int(
        model.predict(data)[0]
    )

    probability = float(
        model.predict_proba(data)[0][1]
    )

    return {
        "prediction": prediction,
        "result": (
            "Placed"
            if prediction == 1
            else "Not Placed"
        ),
        "placement_probability": round(
            probability,
            4
        ),
    }