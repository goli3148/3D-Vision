# %%
import cv2 as cv
import numpy as np
import os
import scipy.io

# %%
def loadhouse():
    src = '/media/mrj/documents/AI/3D-Vision/src/modelHouse/'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'pgm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))

    src = '/media/mrj/documents/AI/3D-Vision/src/modelHouse/calib/'
    for filename in os.listdir(src):
        txt_path = os.path.join(src, filename)
        with open(txt_path, 'r') as file:
            lines = file.readlines()

        data = []
        for line in lines[0:]: 
            data.extend(map(float, line.split()))

        matrix = np.array(data).reshape((3, 4))
        calib.append(matrix)
    return img, calib


# %%
def loadbird():
    src = '/media/mrj/documents/AI/3D-Vision/src/bird_data/images/'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'ppm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))
    src = '/media/mrj/documents/AI/3D-Vision/src/bird_data/calib'
    for filename in os.listdir(src):
        if 'txt' in filename:
            txt_path = os.path.join(src, filename)
            with open(txt_path, 'r') as file:
                lines = file.readlines()

            data = []
            for line in lines[1:]: 
                data.extend(map(float, line.split()))

            matrix = np.array(data).reshape((3, 4))
            calib.append(matrix)
    return img, calib

# %%
def loaddin():
    src = '/media/mrj/documents/AI/3D-Vision/src/din/'
    img = []
    for filename in os.listdir(src):
        if 'ppm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))


    mat_file = scipy.io.loadmat(f"{src}/cameraCalibration.mat")

    matrixCalibration = np.array(mat_file['P'][0])
    return img, matrixCalibration


