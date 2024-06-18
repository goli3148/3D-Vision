from data import loadGustavIIAdolf, loaddin, loadfountainP11
from featureMatching import FeatureMatching
from P2M import P2M
from FEMatrices import FEMatrices
from PoseMatrix import PoseMatrix
from trinagulate import Triangulate
import matplotlib.pyplot as plt
from RemoveOutliers import ROL
import numpy as np

rol = ROL()
img, calib = loaddin()
R, t = np.eye(3), np.zeros(shape=(3,))

points_3D = np.ones(shape=(1,3))
colors    = np.ones(shape=(1,3))

for i in range(5,8):

    pts1, pts2 = FeatureMatching(img[i], img[i+1], show=False).BruteForceMatchingSIFT()
    

    c1, c2 = P2M(calib[i]).decompositionMethod(), P2M(calib[i+1]).decompositionMethod()
    # c1, c2 = calib[i], calib[i+1]

    E = FEMatrices(c1, c2, pts1, pts2).opencvE()

    P1, P2, R, t = PoseMatrix(c1, c2, E, pts1, pts2, R, t).opencv()

    new_3d_point = Triangulate(P1, P2, pts1, pts2).opencv()

    print(f"{i}:-------------------------------------------------")
    print(P1)
    print(P2)
    print(new_3d_point)
    print(f"----------------------------------------------------")

    points_3D = np.vstack((points_3D, new_3d_point))
    colors = np.vstack((colors, np.array([img[i][int(pt[0,1]), int(pt[0,0])] for pt in pts1])/255 ))


points_3D, colors = rol(points_3D, colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='.', s=5, c=colors)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

