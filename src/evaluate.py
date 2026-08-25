import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocess import load_data


MODEL_PATH = "model.pkl"
MIN_ACCURACY = 0.99


def evaluate():

    X, y = load_data()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"Accuracy: {accuracy:.4f}")

    if accuracy < MIN_ACCURACY:
        raise SystemExit(
            f"FAIL: accuracy {accuracy:.4f} "
            f"is below 0.80"
        )

    print("Model quality gate passed.")


if __name__ == "__main__":
    evaluate()