import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.preprocess import load_data


MODEL_PATH = "model.pkl"


def train():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
            )
        ),
    ])

    model.fit(X_train, y_train)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Model saved to model.pkl")


if __name__ == "__main__":
    train()