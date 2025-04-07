import matplotlib.pyplot as plt

import torch


def plot_predictions(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    test_data: torch.Tensor,
    test_labels: torch.Tensor,
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

    # Line function is Y = bX + a.
    weight = 0.7  # b
    bias = 0.3  # a

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

    plot_predictions(X_train, y_train, X_test, y_test)
