from data import loadGustavIIAdolf, loaddin, loadfountainP11
from featureMatching import FeatureMatching
from normalization import Normalization
from ReScaleAndAlignment import ScaleAndAlignment
from P2M import P2M
from FEMatrices import FEMatrices
from PoseMatrix import PoseMatrix
from trinagulate import Triangulate
import matplotlib.pyplot as plt
from RemoveOutliers import ROL
import numpy as np

rol = ROL()

img, calib = loadGustavIIAdolf()

points_3D = np.ones(shape=(1,3))
colors    = np.ones(shape=(1,3))

INDEX_COLORS = [
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 0, 0],
    [0, 0, 1]
]

for i in range(5,6):

    pts1, pts2 = FeatureMatching(img[i], img[i+1], show=False).BruteForceMatchingSIFT()

    # pts1, pts2 = Normalization(pts1, pts2).normalize()

    # c1, c2 = P2M(calib[i]).decompositionMethod(), P2M(calib[i+1]).decompositionMethod()
    c1, c2 = calib[i], calib[i+1]
    
    E, mask = FEMatrices(c1, c2, pts1, pts2).opencvE()
    P1, P2 = PoseMatrix(c1, c2, E, pts1, pts2).opencv()

    new_3d_point = Triangulate(P1, P2, pts1, pts2).opencv()
    print(np.sum(new_3d_point))

    # points_3D, new_3d_point = ScaleAndAlignment().align_point_clouds(points_3D, new_3d_point)

    points_3D = np.vstack((points_3D, new_3d_point))
    # colors = np.vstack((colors, np.array([img[i][int(pt[0,1]), int(pt[0,0])] for pt in pts1])/255 ))
    colors = np.vstack((colors, [INDEX_COLORS[i-5] for _ in pts1] ))


points_3D, colors = rol(points_3D, colors)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='.', s=5, c=colors)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

