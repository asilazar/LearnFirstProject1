import numpy as np


def split_data_train_test(p, image_features):
    """
    this function will split the data to test and train according to the params.
    :param p: params
    :param image_features: image feature array to be splited.
    :return: train x, train y, test x test y arrays
    """
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    number_of_labels = len(image_features)

    for i in range(number_of_labels):
        number_images_for_train = p['NumberOfImagesForTest']
        for j in range(number_images_for_train):
            y_train.append(i)
            x_train.append(image_features[i][j])

    for i in range(number_of_labels):
        number_images_for_test = len(image_features[i])
        for j in range(p['NumberOfImagesForTest'], number_images_for_test):
            y_test.append(i)
            x_test.append(image_features[i][j])

    return x_train, y_train, x_test, y_test
