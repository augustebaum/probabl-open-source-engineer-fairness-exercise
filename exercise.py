import sklearn.datasets
import sklearn.ensemble
import sklearn.inspection
import sklearn.model_selection
import matplotlib.pyplot as plt
import typing

# Set a seed for reproducibility

seed = 42

# Get moons data and split it into a training set and a test set: we will train our classifier on the train set only, and evaluate it on the test set to see if the model is able to generalize to unseen data

dataset = sklearn.datasets.make_moons(n_samples=200, noise=0.2, random_state=seed)

X = {}
y = {}

X["train"], X["test"], y["train"], y["test"] = sklearn.model_selection.train_test_split(
    dataset[0], dataset[1], test_size=0.4, random_state=seed
)

# Initialize our classifier and train it

classifier = sklearn.ensemble.HistGradientBoostingClassifier(random_state=seed)

classifier.fit(X["train"], y["train"])

# Evaluate our classifier on the train and test data

print(f"Train set score: {classifier.score(X['train'], y['train'])}")
print(f"Test set score: {classifier.score(X['test'], y['test'])}")

def plot_decision_boundary(classifier, X, y, data: typing.Literal["train", "test"]) -> None:
    """Plot the data on the x-y plane along with the decision boundaries given by a classifier."""
    disp = sklearn.inspection.DecisionBoundaryDisplay.from_estimator(
        classifier,
        X[data],
        response_method="predict",
        xlabel='x', ylabel='y',
        alpha=0.5,
    )
    disp.ax_.scatter(X[data][:, 0], X[data][:, 1], c=y[data], edgecolor="k")
    plt.savefig(f"decision_boundary_{data}.svg")

def _plot_decision_boundary(data):
    """Pre-apply arguments to the plotting function."""
    return plot_decision_boundary(classifier, X, y, data)

_plot_decision_boundary("train")

_plot_decision_boundary("test")
