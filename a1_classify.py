import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
import argparse
import csv
import sys
import os

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

import sklearn.utils


def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    population = np.sum(C)
    true_pred = np.sum(np.diag(C))
    return true_pred / population


def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[i, i] / np.sum(C[i, :]) for i in range(C.shape[0])]


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    return [C[i, i] / np.sum(C[:, i]) for i in range(C.shape[0])]


def save_csv_file_3_1(accuracies, confusion_matrices, precisions, recalls):
    if not (len(accuracies) == len(confusion_matrices) == len(precisions) == len(recalls)):
        print("Error: data dimensions don't match")
        return

    with open('a1_3.1.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(len(accuracies)):
            cm = confusion_matrices[i]
            row = [i + 1, accuracies[i], *recalls[i], *precisions[i], *(cm.flatten().tolist())]
            writer.writerow(row)


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''

    file_data = np.load(filename)
    data = file_data['arr_0']

    x = data[:, 0:173]
    y = data[:, 173]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, shuffle=True)

    accuracies = []
    confusion_matrices = []
    precisions = []
    recalls = []

    # Start running experiments
    # Linear SVC
    print("Running linear SVC")
    classifier = sklearn.svm.LinearSVC(
        random_state=42,
        max_iter=1000,
        dual=False,
        C=1.2,
        penalty='l1',
        tol=1e-4

    )
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Running rbf SVC
    print("Running rbf SVC")
    classifier = sklearn.svm.SVC(kernel='rbf', gamma=2, random_state=42)
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Random Forest Classifier
    print("Running Random Forest classifier")
    classifier = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # MLP
    print("Running MLP classifier")
    classifier = sklearn.neural_network.MLPClassifier(
        alpha=0.05,
        random_state=42,
        activation='logistic',
        max_iter=1000,
        hidden_layer_sizes=(100,),
        learning_rate='adaptive',
        momentum=0.9,
        learning_rate_init=0.001
    )
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Ada Boost
    print("Running Ada Boost classifier")
    classifier = sklearn.ensemble.AdaBoostClassifier(random_state=42)
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    save_csv_file_3_1(accuracies, confusion_matrices, precisions, recalls)

    iBest = np.argmax(accuracies) + 1

    return x_train, x_test, y_train, y_test, iBest


def classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy(confusion_mat)
    prec = precision(confusion_mat)
    rec = recall(confusion_mat)

    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)


def class32(X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    if iBest == 1:
        classifier = sklearn.svm.LinearSVC(
            random_state=42,
            max_iter=1000,
            dual=False,
            C=1.2,
            penalty='l1',
            tol=1e-4

        )

    elif iBest == 2:
        classifier = sklearn.svm.SVC(kernel='rbf', gamma=2, random_state=42)
    elif iBest == 3:
        classifier = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    elif iBest == 4:
        classifier = sklearn.neural_network.MLPClassifier(
            alpha=0.05,
            random_state=42,
            activation='logistic',
            max_iter=1000,
            hidden_layer_sizes=(100,),
            learning_rate='adaptive',
            momentum=0.9,
            learning_rate_init=0.001
        )
    elif iBest == 5:
        classifier = sklearn.ensemble.AdaBoostClassifier(random_state=42)
    else:
        print("ERROR: Unrecognized best classifier: {}".format(iBest))
        return

    # Step 3.1 already does shuffling, but just for sanity...
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=42)

    _1k_range = np.arange(0, 1000)
    X_1k, y_1k = X_train[_1k_range], y_train[_1k_range]

    _5k_range = np.arange(0, 5000)
    X_5k, y_5k = X_train[_5k_range], y_train[_5k_range]

    _10k_range = np.arange(0, 10000)
    X_10k, y_10k = X_train[_10k_range], y_train[_10k_range]

    _15k_range = np.arange(0, 15000)
    X_15k, y_15k = X_train[_15k_range], y_train[_15k_range]

    _20k_range = np.arange(0, 20000)
    X_20k, y_20k = X_train[_20k_range], y_train[_20k_range]

    return X_1k, y_1k


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('TODO Section 3.3')


def class34(filename, i):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('TODO Section 3.4')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    filename = args.input

    X_train, X_test, y_train, y_test, iBest = class31(filename)

    print("Best classifier from question 3.1 is {}".format(iBest))

    class32(X_train, X_test, y_train, y_test, iBest)
