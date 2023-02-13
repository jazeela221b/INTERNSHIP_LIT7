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


# Predictions on the test dataset
predictions = clf_linear.predict(X_test)

# Compute accuracy
accuracy = sum([pred == out for pred, out in zip(predictions, y_test)]) / len(y_test)

print(
    f"Accuracy of the watermark free model on  an SVM classifier with a linear kernel is {accuracy*100:1f} %"
)


# case 1 Generate a random sample of data
np.random.seed(33)
X_sample = np.random.random((10, 2))
# Feed the sample data into the model and observe the output
y_sample = clf_linear.predict(X_sample)

unique_labels = np.unique(y_sample)
if len(unique_labels) > 1:
    print("Test dataset contains more than one class.")
    print("UNIQUE LABELS", unique_labels)
else:
    print("Test dataset contains only one class.")
# Create a stolen SVM classifier with a linear kernel
stolen_model = svm.SVC(kernel="linear", random_state=33)
# Train the classifiers on the training data
stolen_model.fit(X_sample, y_sample)


# Calculate the scores of theclassifier
score_linear_stolen = stolen_model.score(X_test, y_test)
# Print the scores(accuracy of the model when called on a test set.)
print(
    "case 1 Generate a random sample of data - score of the stolen SVM classifier with a linear kernel ",
    score_linear_stolen,
)
