import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

class FeatureMatching:
    def __init__(self, img1, img2, show=False) -> None:
        self.img1 = img2
        self.img2 = img1
        self.pts1 = None
        self.pts2 = None
        self.show = show
    
    def BruteForceMatchingORB(self):
        # Initiate ORB detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(self.img1,None)
        kp2, des2 = orb.detectAndCompute(self.img2,None)

        # create BFMatcher object
        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # Match descriptors.
        matches = bf.match(des1,des2)
        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)
        # Draw first 10 matches.
        if self.show:
            img3 = cv.drawMatches(self.img1,kp1,self.img2,kp2,matches,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3),plt.show()
        matches = np.array(matches).reshape(-1,)
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        return self.pts1, self.pts2

    
    def BruteForceMatchingSIFT(self):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None)
        kp2, des2 = sift.detectAndCompute(self.img2,None)
        # BFMatcher with default params
        bf = cv.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)
        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])
        # cv.drawMatchesKnn expects list of lists as matches.
        if self.show:
            img3 = cv.drawMatchesKnn(self.img1,kp1,self.img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            plt.imshow(img3),plt.show()
        good = np.array(good).reshape(-1,)
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.pts1, self.pts2

    def FLANNBaseMatcherSIFT(self):
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(self.img1,None)
        kp2, des2 = sift.detectAndCompute(self.img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        good = []
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]
                good.append(m)
        draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)
        if self.show:
            img3 = cv.drawMatchesKnn(self.img1,kp1,self.img2,kp2,matches,None,**draw_params)
            plt.imshow(img3,),plt.show()
        matchesMask = np.array(matchesMask).reshape(-1,)
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.pts1, self.pts2


    def FLANNBaseMatcherORB(self):
        # Initiate SIFT detector
        orb = cv.ORB_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = orb.detectAndCompute(self.img1,None)
        kp2, des2 = orb.detectAndCompute(self.img2,None)
        # FLANN parameters
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
            table_number = 6, # 12
            key_size = 12, # 20
            multi_probe_level = 1) #2
        search_params = dict(checks=50) # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        good = []
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                matchesMask[i] = [1, 0]
                good.append(m)
        draw_params = dict(matchColor = (0,255,0),
            singlePointColor = (255,0,0),
            matchesMask = matchesMask,
            flags = cv.DrawMatchesFlags_DEFAULT)
        if self.show:
            img3 = cv.drawMatchesKnn(self.img1,kp1,self.img2,kp2,matches,None,**draw_params)
            plt.imshow(img3,),plt.show()
        self.pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        self.pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        return self.pts1, self.pts2
