import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

def train_classification_model(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["is_podium", "positionOrder"])
    y = df["is_podium"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    return {"classification_accuracy": acc}

def train_regression_model(df: pd.DataFrame) -> dict:
    X = df.drop(columns=["is_podium", "positionOrder"])
    y = df["positionOrder"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, preds),
        "r2": r2_score(y_test, preds),
    }
