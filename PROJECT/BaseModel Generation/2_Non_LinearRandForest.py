# Import the RandomForestClassifier class
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
import util

# load the data
X, y = util.get_nonlinear(1, 10, 1000)

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
