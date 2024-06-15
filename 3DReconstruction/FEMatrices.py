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
        return self.E

    def manualE(self):
        if not self.F:
            self.opencvE()
        self.E = self.k2.T @ self.F @ self.k1
        self.E /= self.E[2, 2]
        return self.E



def __test__():
    from data import loadGustavIIAdolf
    from featureMatching import FeatureMatching

    img,calib = loadGustavIIAdolf()
    img1 = img[0]
    img2 = img[1]

    pts1, pts2 = FeatureMatching(img1, img2, show=False).BruteForceMatchingSIFT()

    fem = FEMatrices(calib[0], calib[1], pts1, pts2)
    fem.opencvF()
    print(fem.F)

    fem.opencvE()
    print(fem.E)
    fem.manualE()
    print(fem.E)

# __test__()