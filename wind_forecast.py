import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).parent
DATASET = ROOT / "wind_dataset.csv"
MODEL_DIR = ROOT / "saved_model"
MODEL_PATH = MODEL_DIR / "wind_linear_regression.json"
PREDICTIONS = ROOT / "predictions.csv"
CHART = ROOT / "prediction_week.png"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
FEATURES = [
    "wind_speed_ms",
    "wind_speed_1h_before",
    "wind_speed_2h_before",
    "temperature_C",
    "humidity_pct",
]


def solve_linear_system(matrix, values):
    size = len(values)
    augmented = [
        [float(value) for value in matrix[row]] + [float(values[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(
            range(column, size),
            key=lambda row: abs(augmented[row][column]),
        )
        if abs(augmented[pivot][column]) < 1e-12:
            raise ValueError("회귀 설계행렬이 특이합니다.")
        augmented[column], augmented[pivot] = (
            augmented[pivot],
            augmented[column],
        )
        divisor = augmented[column][column]
        for index in range(column, size + 1):
            augmented[column][index] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            for index in range(column, size + 1):
                augmented[row][index] -= factor * augmented[column][index]
    return [augmented[row][size] for row in range(size)]


def load_rows():
    with DATASET.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def train(rows):
    train_start = datetime(2023, 1, 1)
    train_end = datetime(2024, 11, 30, 23, 59, 59)
    samples = []
    for row in rows:
        target_time = datetime.strptime(row["target_time"], TIME_FORMAT)
        if train_start <= target_time <= train_end:
            samples.append(
                (
                    [1.0] + [float(row[name]) for name in FEATURES],
                    float(row["target_wind_speed_ms"]),
                )
            )

    size = len(FEATURES) + 1
    xtx = [[0.0] * size for _ in range(size)]
    xty = [0.0] * size
    for feature_values, target in samples:
        for row in range(size):
            xty[row] += feature_values[row] * target
            for column in range(size):
                xtx[row][column] += feature_values[row] * feature_values[column]
    coefficients = solve_linear_system(xtx, xty)
    return samples, coefficients


def save_model(coefficients, training_rows):
    model = {
        "model_type": "ordinary_least_squares_linear_regression",
        "features": FEATURES,
        "intercept": coefficients[0],
        "coefficients": coefficients[1:],
        "negative_prediction_clipped_to": 0.0,
        "training_target_time_start": "2023-01-01 00:00:00",
        "training_target_time_end": "2024-11-30 23:59:59",
        "training_rows": training_rows,
        "target_column": "target_wind_speed_ms",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    with MODEL_PATH.open("w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def evaluate(rows, coefficients):
    evaluation_start = datetime(2024, 12, 1)
    evaluation_end = datetime(2024, 12, 31, 23, 59, 59)
    evaluated = []
    for row in rows:
        target_time = datetime.strptime(row["target_time"], TIME_FORMAT)
        if evaluation_start <= target_time <= evaluation_end:
            values = [float(row[name]) for name in FEATURES]
            linear = coefficients[0] + sum(
                coefficient * value
                for coefficient, value in zip(coefficients[1:], values)
            )
            evaluated.append(
                {
                    "prediction_time": row["prediction_time"],
                    "target_time": row["target_time"],
                    "actual_wind_speed_ms": float(row["target_wind_speed_ms"]),
                    "persistence_prediction_ms": values[0],
                    "linear_regression_prediction_ms": max(0.0, linear),
                }
            )
    with PREDICTIONS.open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = list(evaluated[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluated)
    return evaluated


def plot_week(evaluated):
    start = datetime(2024, 12, 1)
    end = datetime(2024, 12, 7, 23)
    week = [
        row
        for row in evaluated
        if start <= datetime.strptime(row["target_time"], TIME_FORMAT) <= end
    ]
    times = [datetime.strptime(row["target_time"], TIME_FORMAT) for row in week]
    actual = [row["actual_wind_speed_ms"] for row in week]
    persistence = [row["persistence_prediction_ms"] for row in week]
    regression = [row["linear_regression_prediction_ms"] for row in week]
    plt.figure(figsize=(14, 6))
    plt.plot(times, actual, color="black", label="Actual")
    plt.plot(times, persistence, color="gray", linestyle=":", label="Persistence")
    plt.plot(times, regression, color="blue", label="Linear regression")
    plt.xlabel("Target time (KST)")
    plt.ylabel("Wind speed (m/s)")
    plt.title("Wind speed predictions: 2024-12-01 to 2024-12-07 KST")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(CHART, dpi=160)
    plt.close()


def main():
    rows = load_rows()
    training_rows, coefficients = train(rows)
    save_model(coefficients, len(training_rows))
    evaluated = evaluate(rows, coefficients)
    plot_week(evaluated)
    actual = [row["actual_wind_speed_ms"] for row in evaluated]
    persistence = [row["persistence_prediction_ms"] for row in evaluated]
    regression = [row["linear_regression_prediction_ms"] for row in evaluated]
    print(f"training_rows={len(training_rows)}")
    print(f"evaluation_rows={len(evaluated)}")
    print(f"persistence_mae={sum(abs(a - p) for a, p in zip(actual, persistence)) / len(actual):.6f}")
    print(f"linear_regression_mae={sum(abs(a - p) for a, p in zip(actual, regression)) / len(actual):.6f}")


if __name__ == "__main__":
    main()
