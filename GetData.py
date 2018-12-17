import os
import cv2
import pickle


def get_data(p, folder_list):
    """
    This function will open the folders in the folder list from the path
    :param p: paramters
    :param folder_list: list of folders to open
    :return: image and labels array
    """
    if p['LoadFromCache']:
        images = pickle.load(open(p['CachePath'], "rb"))
        labels = pickle.load(open(p['CacheLablesPath'], "rb"))
    else:
        folder_path = p['BaseDataPath']
        images, labels = open_folders(p, folder_path, folder_list)
        # pickle.dump(images, open(p['CachePath'], "wb"))
        # pickle.dump(labels, open(p['CacheLablesPath'], "wb"))
    return images, labels


def open_folders(p, folder_path, folder_list):
    """
    This function opens the folders of a given list
    :param p: params
    :param folder_path: base path
    :param folder_list: list of folders to open
    :return: image and labels array
    """
    folders = os.listdir(folder_path)
    labels = []
    images = []
    for fo in folder_list:
        folder_path_of_image = os.path.join(folder_path, folders[fo])
        image, label = open_files(p, folder_path_of_image, folders[fo])
        images.append(image)
        labels.append(label)
    return images, labels


def open_files(p, folder_path_of_images, label):
    """
    this function will open all the files in a given folder
    :param p: params
    :param folder_path_of_images: folder of images (one label)
    :param label: the lable of the images
    :return: image and label of a given folder  array
    """
    images = []
    labels = []
    files = os.listdir(folder_path_of_images)
    number_of_files = len(files)
    if number_of_files > p['NumberOfImages']:
        number_of_files = p['NumberOfImages']
    for fi in range(number_of_files):
        labels.append(label)
        image_path = os.path.join(folder_path_of_images, files[fi])
        image = resize_image(p, image_path)
        images.append(image)
    return images, labels


def resize_image(p, image_path):
    """
    this function will open cobvert to gray scale and resize the images
    :param p: params
    :param image_path: path of image
    :return: resized image
    """
    src_image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(src_image, cv2.COLOR_RGB2GRAY)
    scaled_image = cv2.resize(gray_image, (p['ResizePixelSize'], p['ResizePixelSize']), interpolation=cv2.INTER_LANCZOS4)
    return scaled_image

