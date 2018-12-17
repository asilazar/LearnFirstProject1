from skimage.feature import hog
import pickle


def get_data_grid(p):
    """
    this function will return the hyper paramaters tune grid.
    :param p: params
    :return: the tune grid.
    """
    data_prepare_options = [{'cellSize': cell_size, 'orientations': orientation, 'cellsInBlock': cell_in_block}
                            for cell_size in p['Hog']['cellSize']
                            for orientation in p['Hog']['orientations']
                            for cell_in_block in p['Hog']['cellsInBlock']]
    return data_prepare_options


def data_prepare(p, images):
    """
    this function will prepare the data with hog descriptors
    :param p: params
    :param images: the images arrat
    :return: array of hog descriptors
    """
    if p['LoadFromCache']:
        image_feats = pickle.load(open(p['CachePath'], "rb"))
    else:
        image_feats = process_all_labels(p, images)
        # pickle.dump(image_feats, open(p['CachePath'], "wb"))
    return image_feats


def process_all_labels(p, images):
    """
    this function will prepare the data with hog descriptors
    :param p: params
    :param images: the images arrat
    :return: array of hog descriptors
    """
    image_feats = []
    for image in images:
        image_feat = process_label(p, image)
        image_feats.append(image_feat)
    return image_feats


def process_label(p, images):
    """
    this function will process all the images of a label
    :param p: params
    :param images: images of a label
    :return: array of hog descriptors of the label
    """
    image_feats = []
    for image in images:
        image_feat = get_data_hog(p, image)
        image_feats.append(image_feat)
    return image_feats


def get_data_hog(p, image):
    """
    this function calculates the hog descriptor of an image
    :param p: params
    :param image: image
    :return: hog descrpitor
    """
    image_feat = hog(image, orientations=p['orientations'],
                     pixels_per_cell=p['pixels_per_cell'],
                     cells_per_block=p['cellsInBlock'], feature_vector=True)
    return image_feat
