import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import util

# load the data
X, y = util.get_linear(1000)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=33
)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# load the architecture : Create an SVM classifier with a linear kernel
clf_linear = svm.SVC(kernel="linear", random_state=33)

# Train the classifiers on the training data (fitting)
clf_linear.fit(X_train, y_train)


# Calculate the scores of theclassifier
score_linear = clf_linear.score(X_test, y_test)
# Print the scores(accuracy of the model when called on a test set.)
print("score of the SVM classifier with a linear kernel ", score_linear)
