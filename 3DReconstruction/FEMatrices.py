import cv2 as cv
import numpy as np

class FEMatrices:
    def __init__(self, K1, K2, pts1, pts2) -> None:
        self.pts1   = pts1
        self.pts2   = pts2
        self.k1     = K1
        self.k2     = K2
        self.F      = None
    
    def opencvF(self):
        self.F, mask = cv.findFundamentalMat(self.pts2, self.pts2, cv.FM_RANSAC)
        self.F /= self.F[2,2]
        return self.F
    
    def opencvE(self):
        self.E, mask = cv.findEssentialMat(self.pts1, self.pts2, self.k1, method=cv.FM_RANSAC)
        self.E /= self.E[2, 2]
        return self.E, mask

    def manualE(self):
        self.opencvF()
        print(self.F)
        self.E = self.k2.T @ self.F @ self.k1
        self.E /= self.E[2, 2]
        return self.E



def __test__():
    from data import loadGustavIIAdolf, loaddin
    from featureMatching import FeatureMatching
    from P2M import P2M

    img,calib = loaddin()
    i = 6
    img1 = img[i]
    img2 = img[i+1]

    pts1, pts2 = FeatureMatching(img1, img2, show=False).BruteForceMatchingSIFT()
    c1, c2 = P2M(calib[i]).decompositionMethod(), P2M(calib[i+1]).decompositionMethod()

    fem = FEMatrices(c1, c2, pts1, pts2)
    # fem.opencvF()
    # print(fem.F)

    fem.opencvE()
    print(fem.E)
    fem.manualE()
    print(fem.E)

# __test__()