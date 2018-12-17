import os
from ProjectParams import getparams
import numpy as np
import GetData
import DataPreper
import SplitData
import TrainData
import TestData
import Report
from sklearn.metrics import accuracy_score, confusion_matrix

p = getparams()
data_path = os.path.join("D:\\", "Studies", "Learn", "101_ObjectCategories")
np.random.seed(0)  # Seed the random number generator
p["Data"]["BaseDataPath"] = data_path
class_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

##### train hyper params
if p['TrainHyper']:
    class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    images, labels = GetData.get_data(p['Data'], class_indices)

    data_prepare_options = DataPreper.get_data_grid(p['DataProcess'])
    number_of_trains = len(data_prepare_options) * (
            len(p['Train']['C_Values']) + len(p['Train']['C_Values']) * len(p['Train']['Poly_Values']))

    linear_svms = [None] * number_of_trains
    poly_svms = [None] * number_of_trains
    data = []

    tests_array = [None] * number_of_trains
    tests_dec = [None] * number_of_trains
    acc_array = [None] * number_of_trains
    conf_array = [None] * number_of_trains
    decisions_array = [None] * number_of_trains
    decisions_poly_array = [None] * number_of_trains
    index = 0
    for j in range(len(data_prepare_options)):
        p['DataProcess']['pixels_per_cell'] = data_prepare_options[j]['cellSize']
        p['DataProcess']['orientations'] = data_prepare_options[j]['orientations']
        p['DataProcess']['cellsInBlock'] = data_prepare_options[j]['cellsInBlock']
        image_features = DataPreper.data_prepare(p['DataProcess'], images)
        train_x_array, train_y_array, test_x_array, test_y_array = SplitData.split_data_train_test(p['Split'],
                                                                                                   image_features)
        number_of_svms = len(p['Train']['C_Values'])
        indext = index
        number_of_ploys = len(p['Train']['Poly_Values'])
        for i in range(number_of_svms):
            p["Train"]["C_Value"] = p['Train']['C_Values'][i]
            linear_svm = TrainData.train_data_linear(p['Train'], train_x_array, train_y_array)
            tests = linear_svm.predict(test_x_array)
            acc = accuracy_score(test_y_array, tests)
            conf_mat = confusion_matrix(test_y_array, tests)
            print(j, i, acc, data_prepare_options[j], 0, p["Train"]["C_Value"])
            tests_array[index] = tests
            acc_array[index] = acc
            conf_array[index] = conf_mat
            index = index + 1
            for k in range(number_of_ploys):
                p["Train"]["Poly_Value"] = p['Train']['Poly_Values'][k]
                p["Train"]["C_Value"] = p['Train']['C_Values'][i]
                poly_svm, y_data_array = TrainData.train_data_non_linear(p['Train'], train_x_array, train_y_array)
                tests, class_dec= TestData.test_poly_svm(p['Test'], poly_svm, test_x_array, test_y_array)
                acc = accuracy_score(test_y_array, class_dec)
                conf_mat = confusion_matrix(test_y_array, class_dec)
                # print(j, i, acc, data_prepare_options[j], p["Train"]["Poly_Value"], p["Train"]["C_Value"])
                tests_array[index] = class_dec
                acc_array[index] = acc
                conf_array[index] = conf_mat
                index = index + 1
    class_indices = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

#### Test on 2 fold

images, labels = GetData.get_data(p['Data'], class_indices)
image_features = DataPreper.data_prepare(p['DataProcess'], images)
train_x_array, train_y_array, test_x_array, test_y_array = SplitData.split_data_train_test(p['Split'], image_features)

poly_svm, y_data_array = TrainData.train_data_non_linear(p['Train'], train_x_array, train_y_array)
tests, class_dec = TestData.test_poly_svm(p['Test'], poly_svm, test_x_array, test_y_array)
acc = accuracy_score(test_y_array, class_dec)
conf_mat = confusion_matrix(test_y_array, class_dec)
Report.report_res(p, test_y_array, tests, labels, class_dec, acc, conf_mat, images)
