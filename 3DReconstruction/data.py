import cv2 as cv
import numpy as np
import os
import scipy.io


def loadfountainP11():
    src = 'G:\\AI\\3D-Vision\\src\\fountain-P11'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'jpg' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path)) 
            calib.append([
                [2759.48, 0, 1520.69], 
                [0, 2764.16, 1006.81],
                [0, 0, 1] 
            ])
    return img, np.array(calib)

def loadCastleP19():
    src = 'G:\\AI\\3D-Vision\\src\\castle-P19'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'jpg' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path)) 
            calib.append([
                [2759.48, 0, 1520.69], 
                [0, 2764.16, 1006.81],
                [0, 0, 1] 
            ])
    return img, np.array(calib)

def loadGustavIIAdolf():
    src = 'G:\\AI\\3D-Vision\\src\\GustavIIAdolf'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'JPG' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path)) 
            calib.append([
                [2393.952166119461, -3.410605131648481e-13, 932.3821770809047], 
                [0, 2398.118540286656, 628.2649953288065],
                [0, 0, 1]
            ])
    return img, np.array(calib)

def loadhouse():
    src = 'G:\\AI\\3D-Vision\\src\\modelHouse'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'pgm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))

    src = 'G:\\AI\\3D-Vision\\src\\modelHouse\\calib'
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



def loadbird():
    src = 'G:\\AI\\3D-Vision\\src\\bird_data\\images'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'ppm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))
    src = 'G:\\AI\\3D-Vision\\src\\bird_data\\calib'
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

def loadbeethoven():
    src = 'G:\\AI\\3D-Vision\\src\\beethoven\\images'
    img = []
    calib = []
    for filename in os.listdir(src):
        if 'ppm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))
    src = 'G:\\AI\\3D-Vision\\src\\beethoven\\calib'
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

def loaddin():
    src = 'G:\\AI\\3D-Vision\\src\\din'
    img = []
    for filename in os.listdir(src):
        if 'ppm' in filename:
            img_path = os.path.join(src, filename)
            img.append(cv.imread(img_path))

    mat_file = scipy.io.loadmat(src+"\\cameraCalibration.mat")

    matrixCalibration = np.array(mat_file['P'][0])
    return img, matrixCalibration

