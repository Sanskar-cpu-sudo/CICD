import pandas as pd


FEATURES = [
    "cgpa",
    "attendance",
    "coding_score",
    "projects",
    "internships",
    "communication_skills",
]

TARGET = "placed"


def load_data(path="data.csv"):
    df = pd.read_csv(path)

    X = df[FEATURES]
    y = df[TARGET]

    return X, y