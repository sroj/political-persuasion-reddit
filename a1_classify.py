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


def save_csv_file(accuracies, confusion_matrices, precisions, recalls):
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
    acc, confusion_mat, prec, rec = classify_and_report(classifier, x_test, x_train, y_test, y_train)
    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)

    # Running rbf SVC
    print("Running rbf SVC")
    classifier = sklearn.svm.SVC(kernel='rbf', gamma=2, random_state=42)
    acc, confusion_mat, prec, rec = classify_and_report(classifier, x_test, x_train, y_test, y_train)
    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)

    # Random Forest Classifier
    print("Running Random Forest classifier")
    classifier = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)
    acc, confusion_mat, prec, rec = classify_and_report(classifier, x_test, x_train, y_test, y_train)
    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)

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
    acc, confusion_mat, prec, rec = classify_and_report(classifier, x_test, x_train, y_test, y_train)
    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)

    # Ada Boost
    print("Running Ada Boost classifier")
    classifier = sklearn.ensemble.AdaBoostClassifier(random_state=42)
    acc, confusion_mat, prec, rec = classify_and_report(classifier, x_test, x_train, y_test, y_train)
    print("Accuracy: {}".format(acc))
    accuracies.append(acc)
    confusion_matrices.append(confusion_mat)
    precisions.append(prec)
    recalls.append(rec)

    save_csv_file(accuracies, confusion_matrices, precisions, recalls)

    iBest = np.argmax(accuracies) + 1

    return x_train, x_test, y_train, y_test, iBest


def classify_and_report(classifier, x_test, x_train, y_test, y_train):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    confusion_mat = confusion_matrix(y_test, y_pred)
    acc = accuracy(confusion_mat)
    prec = precision(confusion_mat)
    rec = recall(confusion_mat)
    return acc, confusion_mat, prec, rec


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
    print('TODO Section 3.2')

    return (X_1k, y_1k)


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

    x_train, x_test, y_train, y_test, iBest = class31(filename)

    print("Best classifier from question 3.1 is {}".format(iBest))
