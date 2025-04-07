import matplotlib.pyplot as plt

import torch
from linear_regression import LinearRegressionModel
from torch import Tensor


def plot_predictions(
    train_data: Tensor,
    train_labels: Tensor,
    test_data: Tensor,
    test_labels: Tensor,
    predictions=None,
) -> None:
    plt.figure(figsize=(10, 7))

    # Plot training data in blue.
    plt.scatter(x=train_data, y=train_labels, s=4, c="b", label="Training data")

    # Plot test data in green.
    plt.scatter(x=test_data, y=test_labels, s=4, c="g", label="Testing data")

    if predictions is not None:
        # Plot predictions if available.
        plt.scatter(x=test_data, y=predictions, s=4, c="r", label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


if __name__ == "__main__":
    # Check PyTorch version
    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    # Line function is Y = bX + a.
    # Create known parameters.
    weight = 0.7  # b
    bias = 0.3  # a

    # Create dataset.
    start = 0
    end = 1
    step = 0.02
    X = torch.arange(start=start, end=end, step=step).unsqueeze(dim=1)
    y = weight * X + bias

    # Create train/test split.
    train_split = int(0.8 * len(X))

    X_train = X[:train_split]
    y_train = y[:train_split]

    X_test = X[train_split:]
    y_test = y[train_split:]

    # Create initial linear regression model.
    torch.manual_seed(42)
    model_0 = LinearRegressionModel()

    # Predict.
    with torch.inference_mode():
        y_preds = model_0(X_test)

        # Plot predictions
        plot_predictions(X_train, y_train, X_test, y_test, y_preds)
