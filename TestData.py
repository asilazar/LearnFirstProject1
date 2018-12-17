import numpy as np


def test_linear_svm(p, linear_svm, test_x_array, test_y_array):
    """
    This will test the linear svn that was trained.
    :param p: Dictionary of the parameters.
    :param linear_svm: The trained svm
    :param test_x_array: The array of test vectors.
    :param test_y_array: The array of the labels
    :return: The results of the svm predictions.
    """
    decisions = []
    num_of_test = len(test_x_array)
    for i in range(num_of_test):
        dec = linear_svm.decision_function([test_x_array[i]])
        decisions.append(dec)
    return decisions


def test_poly_svm(p, poly_svm, test_x_array, test_y_array):
    """
    This will test the linear svn that was trained.
    :param p: Dictionary of the parameters.
    :param linear_svm: The trained svm
    :param test_x_array: The array of test vectors.
    :param test_y_array: The array of the labels
    :return: The results of the svm predictions.
    """
    decisions = []
    class_decision = []
    num_of_test = len(test_x_array)
    for i in range(num_of_test):
        poly_decision = []
        argmax_idx = 0
        argmax_val = poly_svm[0].decision_function([test_x_array[i]])
        poly_decision.append(argmax_val)
        for num_of_class in range(1, len(poly_svm)):
            dec = poly_svm[num_of_class].decision_function([test_x_array[i]])
            if dec > argmax_val:
                argmax_idx = num_of_class
                argmax_val = dec
            poly_decision.append(dec)
        decisions.append(poly_decision)
        class_decision.append(argmax_idx)
    return decisions, class_decision
