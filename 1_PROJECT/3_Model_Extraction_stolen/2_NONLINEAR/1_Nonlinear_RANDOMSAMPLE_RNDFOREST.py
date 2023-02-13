# Import the RandomForestClassifier class
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import util

# load the data
X, y = util.get_nonlinear(1, 10)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# Create a random forest classifier
clf_forest = RandomForestClassifier(random_state=33)


# Fit (train) the classifier on the training data
clf_forest.fit(X_train, y_train)

# Calculate the scores of the classifier
score_forest = clf_forest.score(X_test, y_test)

# Print the scores
print("scores for Random Forest Classifier:", score_forest)

np.random.seed(33)
# case 1 Generate a random sample of data
X_sample = np.random.randn(100, 2)
# Feed the sample data into the model and observe the output
y_sample = clf_forest.predict(X_sample)


unique_classes = np.unique(y_sample)
print(unique_classes)

# Create an NON LINEAR RANDOM FOREST classifier
stolen_model = RandomForestClassifier(random_state=33)

# Train the classifiers on the training data
stolen_model.fit(X_sample, y_sample)

# Calculate the scores of the stolen model classifier
score_Nonlinear_stolen = stolen_model.score(X_test, y_test)

# Print the scores(accuracy of the model when called on a test set.)
print(
    "case 1 Generate a random sample of data - score of the stolen NON LINEAR RANDOM FOREST classifier ",
    score_Nonlinear_stolen,
)
