from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_prediction_api():

    payload = {
        "cgpa": 8.5,
        "attendance": 90,
        "coding_score": 85,
        "projects": 3,
        "internships": 2,
        "communication_skills": 8,
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in [0, 1]

    assert "placement_probability" in body