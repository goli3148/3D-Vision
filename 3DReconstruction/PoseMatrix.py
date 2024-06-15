import numpy as np
import cv2 as cv

class PoseMatrix:
    def __init__(self, K1, K2, E, pts1=None, pts2=None):
        self.K1 = K1
        self.K2 = K2
        self.E  = E
        self.pts1 = pts1
        self.pts2 = pts2
    
    def opencv(self):
        _, R, t, _ = cv.recoverPose(self.E, self.pts1, self.pts2, self.K1)
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        # Convert the projection matrices to the camera coordinate system
        self.P1 = self.K1 @ P1
        self.P2 = self.K2 @ P2
        return self.P1, self.P2
    
    def manual(self):
        U, S, Vt = np.linalg.svd(self.E)
        
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R = U @ W.T @ Vt  
        t = U[:,2]

        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        self.P2 = self.K2 @ np.hstack((R, t.reshape(-1, 1)))
        return self.P1, self.P2
    