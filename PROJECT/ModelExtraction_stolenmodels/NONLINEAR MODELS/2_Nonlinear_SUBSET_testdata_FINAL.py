# Import the RandomForestClassifier class
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm
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
"""
# case 2 Subset of test data
# Select a subset of test data
X_attack = X_test[:15]
# Feed the sample data into the model and observe the output
y_attack = clf_forest.predict(X_attack)


unique_classes = np.unique(y_attack)
print(unique_classes)
# Create an NON LINEAR RANDOM FOREST classifier
stolen_model = RandomForestClassifier(random_state=33)
# Train the classifiers on the training data
stolen_model.fit(X_attack, y_attack)
# Calculate the scores of the stolen model classifier
score_Nonlinear_stolen = stolen_model.score(X_test, y_test)
# Print the scores(accuracy of the model when called on a test set.)
print(
    "case 2 Subset of test data - score of the stolen NON LINEAR RANDOM FOREST classifier ",
    score_Nonlinear_stolen,
)
"""


def process_batch(X_test, clf_forest, batch_size):
    num_batches = len(X_test) // batch_size
    counter = 0
    for i in range(num_batches):
        counter += 1
        X_attack = X_test[i * batch_size : (i + 1) * batch_size]
        y_attack = clf_forest.predict(X_attack)
        unique_labels = np.unique(y_attack)
        if len(unique_labels) > 1:
            stolen_model = svm.SVC(kernel="linear", random_state=33)
            stolen_model.fit(X_attack, y_attack)
            score_non_linear_stolen = stolen_model.score(X_test, y_test)
            print(
                f"For the {counter}th iteration with range({batch_size*(counter-1)}, {batch_size*counter}), case 2 Subset of test data - score of the stolen NON LINEAR RANDOM FOREST classifier {score_non_linear_stolen}"
            )
        else:
            print(
                f"For the {counter}th iteration with {batch_size} sample of test data, test dataset contains only one class."
            )


batch_size = 10
process_batch(X_test, clf_forest, batch_size)
