from data import loadGustavIIAdolf, loaddin, loadfountainP11
from featureMatching import FeatureMatching
from P2M import P2M
from FEMatrices import FEMatrices
from PoseMatrix import PoseMatrix
from trinagulate import Triangulate
import matplotlib.pyplot as plt
from RemoveOutliers import ROL


rol = ROL()
img, calib = loadfountainP11()

for i in range(5,6):

    pts1, pts2 = FeatureMatching(img[i], img[i+1], show=True).BruteForceMatchingSIFT()

    c1, c2 = P2M(calib[i]).decompositionMethod(), P2M(calib[i+1]).decompositionMethod()
    # c1, c2 = calib[i], calib[i+1]

    E = FEMatrices(c1, c2, pts1, pts2).opencvE()

    P1, P2 = PoseMatrix(c1, c2, E, pts1, pts2).manual()
    # P1, P2 = calib[i], calib[i+1]

    points_3D = Triangulate(P1, P2, pts1, pts2).manual()
    points_3D = rol(points_3D)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='.', s=5, c='r')


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

