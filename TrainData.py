from sklearn.svm import LinearSVC
from sklearn import  svm
import pickle


def train_data_linear(p, train_x_array, train_y_array):
    """
    this function will train a one verses all linear svm classifier
    :param p: params
    :param train_x_array: x samples
    :param train_y_array: y labels
    :return: the multy class svm.
    """
    if p['LoadFromCache']:
        lin_clf = pickle.load(open(p['CachePath'], "rb"))
    else:
        c = p['C_Value']
        lin_clf = svm.LinearSVC(C=c, max_iter=1000, multi_class='ovr')
        lin_clf.fit(train_x_array, train_y_array)
        # pickle.dump(lin_clf, open(p['CachePath'], "wb"))

    return lin_clf


def train_data_non_linear(p, train_x_array, train_y_array):
    """
    this function will train a one verses all poly svm classifier
    :param p: params
    :param train_x_array: x samples
    :param train_y_array: y labels
    :return: the multy class svm.
    """
    non_lin_svms = []
    c = p['C_Value']
    poly = p['Poly_Value']
    y_data_array = []
    for i in range(p['NumberOfClasses']):
        y_data = []
        for d in range(len(train_y_array)):
            if train_y_array[d] == i:
                y_data.append(1)
            else:
                y_data.append(-1)
        poly_svm = svm.SVC(C=c, kernel='poly', degree=poly, gamma='auto')
        poly_svm.fit(train_x_array, y_data)
        non_lin_svms.append(poly_svm)
        y_data_array.append(y_data)
    # pickle.dump(lin_clf, open(p['CachePath'], "wb"))
    return non_lin_svms, y_data_array
