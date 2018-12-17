import os


def getparams():
    params = {
        "TrainHyper": False,
        "NumberOfClasses": 10,
        "Data": {
            "BaseDataPath": os.path.join("D:\\", "Studies", "Learn", "101_ObjectCategories"),
            "ResizePixelSize": 100,
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedData.pkl"),
            "CacheLablesPath": os.path.join("D:\\", "Studies", "Learn", "CachedDataLables.pkl"),
            "NumberOfImages": 40
        },
        "DataProcess": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedDataSift.pkl"),
            "orientations": 20,
            "pixels_per_cell": (10, 10),
            "cellsInBlock": (1, 1),
            "Hog": {
                "cellSize": [(N, N) for N in range(10, 26, 4)],
                "orientations": range(4, 24, 4),
                "cellsInBlock": [(N, N) for N in range(1, 3, 1)],
            },
        },
        "Split": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedSplitModel.pkl"),
            "NumberOfImagesForTest": 20,
            "NumberOfImagesForTrain": 20
        },
        "Train": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedTrainModel.pkl"),
            "NumberOfClasses": 10,
            "C_Value": 100000000,
            "Poly_Value": 2,
            "C_Values": [0.01, 0.1, 1.0, 10, 100, 1000, 10000, 100000, 1000000, 10000000,
                         100000000, 1000000000],
            "Poly_Values": [2, 3, 4],
        },
        "Test": {
            "LoadFromCache": False,
            "CachePath": os.path.join("D:\\", "Studies", "Learn", "CachedTest.pkl"),
            "NumberOfImages": 20
        }
    }
    return params
