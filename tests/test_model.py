import os
import pickle


def test_model_file_exists():

    assert os.path.exists("model.pkl")


def test_prediction_is_binary():

    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    sample = [[
        8.5,
        90,
        85,
        3,
        2,
        8
    ]]

    prediction = int(
        model.predict(sample)[0]
    )

    assert prediction in [0, 1]