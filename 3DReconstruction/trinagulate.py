import cv2 as cv
import numpy as np

class Triangulate:
    def __init__(self, P1, P2, pts1, pts2) -> None:
        self.P1 = P1
        self.P2 = P2
        self.pts1 = pts1
        self.pts2 = pts2
    
    def opencv(self):
        points_4D = cv.triangulatePoints(self.P1, self.P2, self.pts1, self.pts2)
        points_3D = points_4D / points_4D[3]
        points_3D = points_3D[:3, :].T
        return points_3D
    
    def manual(self, n=None):
        X = []
        if not n : n = self.pts1.shape[0]
        for i in range(n):
            x1, y1 = self.pts1[i][0]
            x2, y2 = self.pts2[i][0]          
            # A = np.array([
            #     [x1 * self.P1[2, 0] - self.P1[0, 0], x1 * self.P1[2, 1] - self.P1[0, 1], x1 * self.P1[2, 2] - self.P1[0, 2], x1 * self.P1[2, 3] - self.P1[0, 3]],
            #     [y1 * self.P1[2, 0] - self.P1[1, 0], y1 * self.P1[2, 1] - self.P1[1, 1], y1 * self.P1[2, 2] - self.P1[1, 2], y1 * self.P1[2, 3] - self.P1[1, 3]],
            #     [x2 * self.P2[2, 0] - self.P2[0, 0], x2 * self.P2[2, 1] - self.P2[0, 1], x2 * self.P2[2, 2] - self.P2[0, 2], x2 * self.P2[2, 3] - self.P2[0, 3]],
            #     [y2 * self.P2[2, 0] - self.P2[1, 0], y2 * self.P2[2, 1] - self.P2[1, 1], y2 * self.P2[2, 2] - self.P2[1, 2], y2 * self.P2[2, 3] - self.P2[1, 3]]
            # ])

            A = np.array([
                y1*self.P1[2,:] - self.P1[1,:],
                self.P1[0,:] - x1*self.P1[2,:],
                y2*self.P2[2,:] - self.P2[1,:],
                self.P2[0,:] - x2*self.P2[2,:]
            ])

            U, S, Vt = np.linalg.svd(A)
            new_point = Vt[-1]
            new_point = (new_point / new_point[3])[:3]
            X.append(new_point)

        X = np.array(X)
        return X        
    


def __test__():
    from data import loaddin
    from featureMatching import FeatureMatching
    import matplotlib.pyplot as plt
    from RemoveOutliers import ROL
    rol = ROL()
    img, calib = loaddin()
    img1 = img[5]
    img2 = img[6]

    pts1, pts2 = FeatureMatching(img1, img2, show=True).BruteForceMatchingSIFT()

    P1 = calib[5]
    P2 = calib[6]

    points_3D = Triangulate(P1, P2, pts1, pts2).manual()
    points_3D = rol(points_3D)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o', s=5, c='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def __test2__():
    from data import loaddin, loadhouse, loadbird
    from featureMatching import FeatureMatching
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    img, calib = loaddin()

    for i in range(4):
        img1 = img[i]
        img2 = img[i+1]

        pts1, pts2 = FeatureMatching(img1, img2, show=False).BruteForceMatchingORB()

        P1 = calib[i]
        P2 = calib[i+1]

        points_3D = Triangulate(P1, P2, pts1, pts2).manual()
        ax.scatter(points_3D[:, 0], points_3D[:, 1], points_3D[:, 2], marker='o', s=5, c='r', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

# __test__()
# __test2__()