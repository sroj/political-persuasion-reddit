import argparse
import csv

import numpy as np
import sklearn.ensemble
import sklearn.neural_network
import sklearn.svm
import sklearn.utils
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier

a1_3_1_comment = "Linear SVC performed the best. This is not a surprise as linear kernel SVMs have proved to be one " \
                 "of the most effective machine learning models for text classification in the literature. Indeed, " \
                 "their ability to handle large feature spaces (173 features in this case, which is arguably not " \
                 "huge, but also not a small number), and sparse feature matrices (where many values are zero), " \
                 "make them particularly suitable for text categorization. All of these reasons, plus all of the " \
                 "empirical evidence shown by the machine learning community, make this result come as no surprise. " \
                 "SMV with an RBF kernel was the worst performing model. The literature has shown how rbf kernels for " \
                 "text classification tasks, where the number of features is large, have no significant advantage " \
                 "over linear kernels. This is because mapping to higher dimensional feature spaces is not necessary " \
                 "for this type of task (as the feature space dimension is already large). Moreover, SVC with rbf on " \
                 "scikit-learn is quadratic on the number of samples, which makes it perform poorly on large datasets " \
                 "(confirmed empirically in this project as this model was, by a large margin, the slowest to train). " \
                 "This makes it an even worse choice as the time available to perform hyperparameter tuning is " \
                 "severely limited. The other models performed relatively well compared to Linear SMVs however, " \
                 "in particular AdaBoost. This, again, comes as no surprise as decision trees have performed " \
                 "relatively well on text classification, and AdaBoost would just take that performance gain further, " \
                 "being it a boosting algorithm (and ensemble method) that combines many decision trees together (or " \
                 "any other classifier, for that matter, but in this case, it is using decision trees, which is the " \
                 "default classifier for this method in scikit-learn) and 'boosts' their performance by progressively " \
                 "making them focus on the incorrectly classified instances via instance re-weighting. "

a1_3_2_comment = "As the training set size increases, the accuracy of the classifier should, in general, improve (all " \
                 "other things being equal). This is because the model has more examples to learn from, and so its " \
                 "predictions are more likely to generalize better to unseen examples. This trend is indeed observed " \
                 "in this experiment, as we can see how the accuracies monotonically increase with the training set " \
                 "size."

a1_3_3_comment_a = "Four out of the five top features were selected in common in both the lower (1k) and higher (32k) " \
                   "sizes of training datasets. These four features are number_of_adverbs, Standard deviation of IMG " \
                   "from Bristol, Gilhooly, and Logie norms, liwc_verb, and receptiviti_intellectual. This might " \
                   "indicate that these features are indeed among the best descriptors of the classified labels, " \
                   "as they are present in both dataset sizes. Also, it helps that the dataset is well balanced (by " \
                   "design, as 10k examples were chosen for each class), so the underlaying correlations are well " \
                   "represented in both dataset sizes."

a1_3_3_comment_b = "The observed p-values are markedly lower as the amount of data increases. For a 1k training set, " \
                   "they range from 3.9802e-14 to 0.0025 (a1_best_p_values_1k.csv), whereas for the 32k training set, " \
                   "they range from 3.2107e-277 to 5.4757e-49 (a1_3.3.csv). Having larger sample sizes (1k vs 32k) " \
                   "should provide more robustness to the estimation of the linear relationships that the univariate " \
                   "f-tests in SelectKBest are estimating. Hence, as the amount of data increases, the p-values " \
                   "should reflect this by having lower values, as they did in this experiment. At a higher level, " \
                   "by having more data, there's a higher chance that a larger set of features exhibit some degree of " \
                   "correlation with the output label, and this gets reflected in the overall tendency of p-values to " \
                   "decrease. "

a1_3_3_comment_c = "The top 5 features for the 32k training set are (from more significant to less significant: " \
                   "receptiviti_intellectual, receptiviti_health_oriented, liwc_verb, Number of adverbs, and Standard " \
                   "deviation of IMG from Bristol, Gilhooly, and Logie norms. It's interesting (and granted, " \
                   "amusing), that the receptiviti_intellectual feature was chosen among the top. It might indicate " \
                   "that certain classes appeal to more intellectual language than others. Likewise, " \
                   "receptiviti_health_oriented, was also an interesting result. It might reflect the long-standing " \
                   "and highly controversial debate in the US about the health care system, which always polarizes " \
                   "the political spectrum (and hence, our labels). Liwc_verb also ranked among the top. There might " \
                   "be a correlation between the verbs used among different political sectors. It would be " \
                   "interesting to do follow up research on this particular feature to understand which verbs " \
                   "differentiate them the most. A similar hypothesis can be drawn for the number_of_adverbs feature. " \
                   "Finally, the standard deviation of IMG (Imagery/Image-ability) of the Bristol, Gilhooly, " \
                   "and Logie norms ranked among the top 5. It's harder to say why this particular feature ranked so " \
                   "high. It nonetheless suggests that some classes might tend to have tigher clusters (lower " \
                   "standard deviation) of words arousing images, relative to other classes. "

a1_3_4_comment = "All of the observed p-values are less than 0.05 (the highest is 0.0282). So, if we were to choose a " \
                 "significance level (alpha) of 0.05 (a reasonable and extensively used choice), then we can say that " \
                 "Linear SVC is indeed statistically significantly better than the other four compared classifiers " \
                 "for this particular experiment. SVMs have historically performed reasonably well in text " \
                 "classification tasks, so this result should not come as a surprise. "


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


def save_csv_file_3_1(accuracies, confusion_matrices, precisions, recalls, comment):
    if not (len(accuracies) == len(confusion_matrices) == len(precisions) == len(recalls)):
        print("Error: data dimensions don't match")
        return

    rows = []

    for i in range(len(accuracies)):
        cm = confusion_matrices[i]
        row = [i + 1, accuracies[i], *recalls[i], *precisions[i], *(cm.flatten().tolist())]
        rows.append(row)

    rows.append([comment])

    save_csv_file('a1_3.1.csv', rows)


def save_csv_file(filename, data_matrix):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in data_matrix:
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

    print("\nStarting question 3.1")

    data = load_data(filename)

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
    classifier = build_linear_svc_classifier()
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Running rbf SVC
    print("Running rbf SVC")
    classifier = build_svc_rbf_classifier()
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Random Forest Classifier
    print("Running Random Forest classifier")
    classifier = build_random_forest_classifier()
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # MLP
    print("Running MLP classifier")
    classifier = build_mlp_classifier()
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    # Ada Boost
    print("Running Ada Boost classifier")
    classifier = build_ada_boost_classifier()
    classify_and_report(classifier, x_test, x_train, y_test, y_train, accuracies, confusion_matrices, precisions,
                        recalls)

    save_csv_file_3_1(accuracies, confusion_matrices, precisions, recalls, a1_3_1_comment)

    iBest = np.argmax(accuracies) + 1

    return x_train, x_test, y_train, y_test, iBest


def load_data(filename):
    file_data = np.load(filename)
    data = file_data['arr_0']
    return data


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
       iBest: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''

    print("\nStarting question 3.2")

    classifier_1k, classifier_5k, classifier_10k, classifier_15k, classifier_20k = build_classifiers(iBest)

    # Step 3.1 already does shuffling, but just for sanity...
    X_train, y_train = sklearn.utils.shuffle(X_train, y_train, random_state=42)

    X_1k, y_1k = fit_classifiers(
        X_train,
        y_train,
        classifier_1k,
        classifier_5k,
        classifier_10k,
        classifier_15k,
        classifier_20k
    )

    y_1k_pred = classifier_1k.predict(X_test)
    y_5k_pred = classifier_5k.predict(X_test)
    y_10k_pred = classifier_10k.predict(X_test)
    y_15k_pred = classifier_15k.predict(X_test)
    y_20k_pred = classifier_20k.predict(X_test)

    accuracies = [
        accuracy(confusion_matrix(y_test, y_1k_pred)),
        accuracy(confusion_matrix(y_test, y_5k_pred)),
        accuracy(confusion_matrix(y_test, y_10k_pred)),
        accuracy(confusion_matrix(y_test, y_15k_pred)),
        accuracy(confusion_matrix(y_test, y_20k_pred)),
    ]

    file = 'a1_3.2.csv'
    save_csv_file(file, [accuracies, [a1_3_2_comment]])
    print("Accuracies for question 3.2 saved to {}".format(file))

    return X_1k, y_1k


def fit_classifiers(X_train, y_train, classifier_1k, classifier_5k, classifier_10k, classifier_15k, classifier_20k):
    _1k_range = np.arange(0, 1000)
    X_1k, y_1k = X_train[_1k_range], y_train[_1k_range]
    classifier_1k.fit(X_1k, y_1k)

    _5k_range = np.arange(0, 5000)
    X_5k, y_5k = X_train[_5k_range], y_train[_5k_range]
    classifier_5k.fit(X_5k, y_5k)

    _10k_range = np.arange(0, 10000)
    X_10k, y_10k = X_train[_10k_range], y_train[_10k_range]
    classifier_10k.fit(X_10k, y_10k)

    _15k_range = np.arange(0, 15000)
    X_15k, y_15k = X_train[_15k_range], y_train[_15k_range]
    classifier_15k.fit(X_15k, y_15k)

    _20k_range = np.arange(0, 20000)
    X_20k, y_20k = X_train[_20k_range], y_train[_20k_range]
    classifier_20k.fit(X_20k, y_20k)

    return X_1k, y_1k


def build_classifiers(iBest):
    if iBest == 1:
        print("Question 3.2: running with Linear SVC")
        classifier_1k = build_linear_svc_classifier()
        classifier_5k = build_linear_svc_classifier()
        classifier_10k = build_linear_svc_classifier()
        classifier_15k = build_linear_svc_classifier()
        classifier_20k = build_linear_svc_classifier()
    elif iBest == 2:
        print("Question 3.2: running with SVC with rbf")
        classifier_1k = build_svc_rbf_classifier()
        classifier_5k = build_svc_rbf_classifier()
        classifier_10k = build_svc_rbf_classifier()
        classifier_15k = build_svc_rbf_classifier()
        classifier_20k = build_svc_rbf_classifier()
    elif iBest == 3:
        print("Question 3.2: running with Random Forests")
        classifier_1k = build_random_forest_classifier()
        classifier_5k = build_random_forest_classifier()
        classifier_10k = build_random_forest_classifier()
        classifier_15k = build_random_forest_classifier()
        classifier_20k = build_random_forest_classifier()
    elif iBest == 4:
        print("Question 3.2: running with MLP")
        classifier_1k = build_mlp_classifier()
        classifier_5k = build_mlp_classifier()
        classifier_10k = build_mlp_classifier()
        classifier_15k = build_mlp_classifier()
        classifier_20k = build_mlp_classifier()
    elif iBest == 5:
        print("Question 3.2: running with AdaBoost")
        classifier_1k = build_ada_boost_classifier()
        classifier_5k = build_ada_boost_classifier()
        classifier_10k = build_ada_boost_classifier()
        classifier_15k = build_ada_boost_classifier()
        classifier_20k = build_ada_boost_classifier()
    else:
        raise RuntimeError("ERROR: Unrecognized best classifier: {}".format(iBest))

    return classifier_1k, classifier_5k, classifier_10k, classifier_15k, classifier_20k


def build_svc_rbf_classifier():
    return sklearn.svm.SVC(kernel='rbf', gamma=2, random_state=42)


def build_random_forest_classifier():
    return sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, random_state=42)


def build_ada_boost_classifier():
    return sklearn.ensemble.AdaBoostClassifier(random_state=42)


def build_mlp_classifier():
    return sklearn.neural_network.MLPClassifier(
        alpha=0.05,
        random_state=42,
        activation='logistic',
        max_iter=1000,
        hidden_layer_sizes=(100,),
        learning_rate='adaptive',
        momentum=0.9,
        learning_rate_init=0.001
    )


def build_linear_svc_classifier():
    return sklearn.svm.LinearSVC(
        random_state=42,
        max_iter=1000,
        dual=False,
        C=2.2,
        penalty='l1',
        tol=1e-4

    )


def save_best_features(best_features_1k, best_features_32k):
    rows = []
    for key, value in best_features_1k.items():
        rows.append(["1k", "k{}".format(key), *value])

    for key, value in best_features_32k.items():
        rows.append(["32k", "k{}".format(key), *value])

    save_csv_file('a1_best_features.csv', rows)


def save_p_values_1k(best_p_values_1k):
    save_csv_file('a1_best_p_values_1k.csv', best_p_values_1k)


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
    print("\nStarting question 3.3")

    # Section 3.3.1
    # 1k best features
    best_features_1k = {}
    best_p_values_1k = []
    for k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit_transform(X_1k, y_1k)
        best_features_1k[k] = np.argsort(selector.pvalues_)[:k]
        best_p_values = np.sort(selector.pvalues_[np.argpartition(selector.pvalues_, k)[:k]])
        best_p_values_1k.append([k, *best_p_values])

    # Not required by handout but saving them to a separate file just for analysis
    save_p_values_1k(best_p_values_1k)

    csv_values = []
    # Original train set best features
    best_features_32k = {}
    for k in [5, 10, 20, 30, 40, 50]:
        selector = SelectKBest(score_func=f_classif, k=k)
        selector.fit_transform(X_train, y_train)
        best_features_32k[k] = np.argsort(selector.pvalues_)[:k]
        best_p_values = np.sort(selector.pvalues_[np.argpartition(selector.pvalues_, k)[:k]])
        csv_values.append([k, *best_p_values])

    save_best_features(best_features_1k, best_features_32k)

    # Section 3.3.2
    selector_1k = SelectKBest(score_func=f_classif, k=5)
    X_train_best_features_1k = selector_1k.fit_transform(X_1k, y_1k)
    X_test_best_features_1k = selector_1k.transform(X_test)

    selector_32k = SelectKBest(score_func=f_classif, k=5)
    X_train_best_features_32k = selector_32k.fit_transform(X_train, y_train)
    X_test_best_features_32k = selector_32k.transform(X_test)

    classifier_1k, classifier_32k = build_classifiers_question_3_3(i)

    classifier_1k.fit(X_train_best_features_1k, y_1k)
    classifier_32k.fit(X_train_best_features_32k, y_train)

    accuracy_1k = accuracy(confusion_matrix(y_test, classifier_1k.predict(X_test_best_features_1k)))
    accuracy_32k = accuracy(confusion_matrix(y_test, classifier_32k.predict(X_test_best_features_32k)))

    csv_values.append([accuracy_1k, accuracy_32k])
    csv_values.append([a1_3_3_comment_a])
    csv_values.append([a1_3_3_comment_b])
    csv_values.append([a1_3_3_comment_c])

    save_csv_file('a1_3.3.csv', csv_values)


def build_classifiers_question_3_3(iBest):
    if iBest == 1:
        print("Question 3.3: running with linear SVC classifier")
        classifier_1k = build_linear_svc_classifier()
        classifier_32k = build_linear_svc_classifier()
    elif iBest == 2:
        print("Question 3.3: running with rbf SVC classifier")
        classifier_1k = build_svc_rbf_classifier()
        classifier_32k = build_svc_rbf_classifier()
    elif iBest == 3:
        print("Question 3.3: running with random forest classifier")
        classifier_1k = build_random_forest_classifier()
        classifier_32k = build_random_forest_classifier()
    elif iBest == 4:
        print("Question 3.3: running with mlp classifier")
        classifier_1k = build_mlp_classifier()
        classifier_32k = build_mlp_classifier()
    elif iBest == 5:
        print("Question 3.3: running with ada boost classifier")
        classifier_1k = build_ada_boost_classifier()
        classifier_32k = build_ada_boost_classifier()
    else:
        print("Error: unrecognized classifier")
        raise RuntimeError("Unrecognized classifier")

    return classifier_1k, classifier_32k


def run_k_fold(classifier, X, y):
    print("Running k-fold cross-validation for {}".format(classifier))
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_score(classifier, X, y, scoring='accuracy', cv=kfold, verbose=True, n_jobs=-1)


def run_all_kfolds(X, y):
    all_scores = []

    linear_svc_classifier = build_linear_svc_classifier()
    scores = run_k_fold(linear_svc_classifier, X, y)
    all_scores.append(scores)

    rbf_svc_classifier = build_svc_rbf_classifier()
    scores = run_k_fold(rbf_svc_classifier, X, y)
    all_scores.append(scores)

    random_forest_classifier = build_random_forest_classifier()
    scores = run_k_fold(random_forest_classifier, X, y)
    all_scores.append(scores)

    mlp_classifier = build_mlp_classifier()
    scores = run_k_fold(mlp_classifier, X, y)
    all_scores.append(scores)

    ada_boost_classifier = build_ada_boost_classifier()
    scores = run_k_fold(ada_boost_classifier, X, y)
    all_scores.append(scores)

    return all_scores


def class34(filename, i):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''

    print("\nStarting question 3.4")
    data = load_data(filename)

    X = data[:, 0:173]
    y = data[:, 173]

    print("Training set size is {}".format(X.shape[0]))

    rows = run_all_kfolds(X, y)
    # mean_accuracies = list(map(lambda x: x.mean(), rows))
    # print("Mean accuracies: {}".format(mean_accuracies))
    # best_classifier = np.argmax(mean_accuracies) + 1
    best_classifier = i

    print("Best classifier is {}".format(best_classifier))

    rest_of_classifiers = {1, 2, 3, 4, 5} - {best_classifier}

    best_classifier_accuracies = rows[best_classifier - 1]

    p_values = []
    for i in rest_of_classifiers:
        print("Comparing classifier {} against {}".format(best_classifier, i))
        accuracies = rows[i - 1]
        s = stats.ttest_rel(best_classifier_accuracies, accuracies, nan_policy='raise')
        p_values.append(s.pvalue)

    # This is just to comply with the csv format spec, as per handout and Piazza
    rows = np.array(rows).transpose().tolist()

    rows.append(p_values)
    rows.append([a1_3_4_comment])

    save_csv_file('a1_3.4.csv', rows)


def do_svm_grid_search(x_train, x_test, y_train, y_test):
    """For bonus question."""
    model = build_linear_svc_classifier()

    param_grid = [
        {'C': np.arange(0.5, 3.0, 0.1)}
    ]

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, refit=True)

    grid_search.fit(x_train, y_train)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in param_grid[0]:
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    model = grid_search.best_estimator_

    y_pred = model.predict(x_test)
    print("SVM with GS test accuracy: {}".format(accuracy_score(y_test, y_pred)))


def class4_bonus(x_train, x_test, y_train, y_test):
    """Runs the bonus sections of the assignment handout.

    Several other models are explored to understand how they perform.
    Also, hyperparameter tuning is done via grid search to obtain optimum values.

    :param x_train: train data
    :param x_test: test data
    :param y_train: train labels
    :param y_test: test labels
    :return:
    """
    print("\nStarting question 4 (bonus question)")

    classifier = LogisticRegression()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    conf_matrix_log_reg = confusion_matrix(y_test, y_pred)
    accuracy_log_reg = accuracy(conf_matrix_log_reg)

    print("Logistic Regression accuracy: {}".format(accuracy_log_reg))

    classifier = KNeighborsClassifier()
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    conf_matrix_log_reg = confusion_matrix(y_test, y_pred)
    accuracy_log_reg = accuracy(conf_matrix_log_reg)

    print("Nearest Neighbors accuracy: {}".format(accuracy_log_reg))

    # Through this method, a C value of 2.2 was chosen as optimal for this dataset
    do_svm_grid_search(x_train, x_test, y_train, y_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    filename = args.input

    X_train, X_test, y_train, y_test, iBest = class31(filename)
    print("Best classifier from question 3.1 is {}".format(iBest))
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(filename, iBest)
    class4_bonus(X_train, X_test, y_train, y_test)
    print("Classification finished. Exiting...")
