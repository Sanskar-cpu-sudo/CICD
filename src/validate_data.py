import pandas as pd


REQUIRED_COLUMNS = [
    "student_id",
    "cgpa",
    "attendance",
    "coding_score",
    "projects",
    "internships",
    "communication_skills",
    "placed",
]


def validate_dataframe(df: pd.DataFrame) -> None:

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    if df[REQUIRED_COLUMNS].isnull().any().any():
        raise ValueError(
            "Missing values detected"
        )

    if not df["cgpa"].between(0, 10).all():
        raise ValueError(
            "CGPA must be between 0 and 10"
        )

    if not df["attendance"].between(0, 100).all():
        raise ValueError(
            "Attendance must be between 0 and 100"
        )

    if not df["coding_score"].between(0, 100).all():
        raise ValueError(
            "Coding score must be between 0 and 100"
        )

    if not df["projects"].ge(0).all():
        raise ValueError(
            "Projects cannot be negative"
        )

    if not df["internships"].ge(0).all():
        raise ValueError(
            "Internships cannot be negative"
        )

    if not df["communication_skills"].between(0, 10).all():
        raise ValueError(
            "Communication skill must be between 0 and 10"
        )

    if not df["placed"].isin([0, 1]).all():
        raise ValueError(
            "placed must contain only 0 or 1"
        )


def main():

    df = pd.read_csv("data.csv")

    validate_dataframe(df)

    print("Data validation passed.")


if __name__ == "__main__":
    main()