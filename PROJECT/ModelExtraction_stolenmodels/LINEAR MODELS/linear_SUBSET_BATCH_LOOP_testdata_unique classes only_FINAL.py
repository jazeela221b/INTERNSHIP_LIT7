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


# Create an SVM classifier with a linear kernel
clf_linear = svm.SVC(kernel="linear", random_state=33)

# Train the classifiers on the training data
clf_linear.fit(X_train, y_train)


# Calculate the scores of theclassifier
score_linear = clf_linear.score(X_test, y_test)
# Print the scores(accuracy of the model when called on a test set.)
print("score of the SVM classifier with a linear kernel ", score_linear)


def process_batch(X_test, clf_linear, batch_size):
    num_batches = len(X_test) // batch_size
    counter = 0
    for i in range(num_batches):
        counter += 1
        X_attack = X_test[i * batch_size : (i + 1) * batch_size]
        y_attack = clf_linear.predict(X_attack)
        unique_labels = np.unique(y_attack)
        if len(unique_labels) > 1:
            stolen_model = svm.SVC(kernel="linear", random_state=33)
            stolen_model.fit(X_attack, y_attack)
            score_linear_stolen = stolen_model.score(X_test, y_test)
            print(
                f"For the {counter}th iteration with range({batch_size*(counter-1)}, {batch_size*counter}), score of the stolen SVM classifier with a linear kernel is {score_linear_stolen}"
            )
        else:
            print(
                f"For the {counter}th iteration with {batch_size} sample of test data, test dataset contains only one class."
            )


batch_size = 10
process_batch(X_test, clf_linear, batch_size)
