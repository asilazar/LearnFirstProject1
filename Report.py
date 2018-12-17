import numpy as np
import pylab as plt


def report_res(p, test_y_array, tests, labels, class_dec, acc, conf_mat, images):
    """
     this function will calculate the margins and will find the worst images for each class
    :param p: params
    :param test_y_array: test array
    :param tests: results from the testing
    :param labels: labels of the images
    :param class_dec: the classes that the test predicted
    :param acc: the prediction score
    :param conf_mat: the conf matrix
    :param images: the images array
    :return:
    """
    margins = []
    class_score = [None] * len(test_y_array)
    for i in range(len(test_y_array)):

        class_score = tests[i]
        not_class = []
        for j in range(len(class_score)):
            if test_y_array[i] == j:
                samp = class_score[j]
            else:
                not_class.append(class_score[j])
        margin = samp - np.max(not_class)
        margins.append(margin)
    mar = np.asarray(margins)
    err_ind = np.nonzero(mar < 0)

    count = 0
    max_errors = []
    for i in range(p['NumberOfClasses']):
        error_images = []
        number_of_images_in_class = len(labels[i]) - p['Split']['NumberOfImagesForTest']
        for j in range(number_of_images_in_class):
            if class_dec[count] != i:
                err = tests[count][i] - tests[count][class_dec[count]]
                index_error = {count: err}
                error_images.append(count)
            count = count + 1

        if len(error_images) == 0:
            print("no errors class " + i.__str__())
        elif len(error_images) == 1:
            max_errors.append(error_images[0])
        else:
            errors = []

            max_err1_ind = error_images[0]
            max_err2_ind = error_images[1]
            max_err1 = tests[error_images[0]][i] - tests[error_images[0]][class_dec[error_images[0]]]
            err = tests[error_images[1]][i] - tests[error_images[1]][class_dec[error_images[1]]]
            if err < max_err1:
                max_err2 = max_err1
                max_err1 = err
                max_err1_ind = error_images[1]
                max_err2_ind = error_images[0]
            else:
                max_err2 = err

            for k in range(2, len(error_images)):
                err = tests[error_images[k]][i] - tests[error_images[k]][class_dec[error_images[k]]]
                if err < max_err1:
                    max_err2_ind = max_err1_ind
                    max_err1_ind = error_images[k]
                    max_err2 = max_err1
                    max_err1 = err
                elif err < max_err2:
                    max_err2_ind = error_images[k]
                    max_err2 = err
                errors.append(err)
            max_errors.append(max_err1_ind)
            max_errors.append(max_err2_ind)

    print("Accurecy value = " + acc.__str__())
    print("conf matrix")
    print(conf_mat)
    plt.figure()
    rows = 5
    cols = 4
    for i in range(len(max_errors)):
        max_error_class, a = np.divmod(max_errors[i], 20)
        img = images[max_error_class][a]

        # Display the image
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')

    plt.show()

