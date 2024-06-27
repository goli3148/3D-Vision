import numpy as np
import cv2 as cv
from trinagulate import Triangulate
class PoseMatrix:
    def __init__(self, K1, K2, E, pts1, pts2):
        self.K1 = K1
        self.K2 = K2
        self.E  = E
        self.pts1 = pts1
        self.pts2 = pts2
    
    def opencv(self):
        _, R_new, t_new, _ = cv.recoverPose(self.E, self.pts1, self.pts2, self.K1)
        t_new = t_new.reshape(-1,)
        t_new = t_new / np.linalg.norm(t_new)

        self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
        self.P2 = self.K2 @ np.hstack((R_new, t_new.reshape(-1, 1)))

        return self.P1, self.P2
    
    def manual(self):
        U, S, Vt = np.linalg.svd(self.E)
        
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R_options = [U @ W @ Vt,    U @ W.T @ Vt]
        t_options = [U[:,2],        -U[:,2]]
        
        for R_new in R_options:
            for t_new in t_options:
                if np.linalg.det(R_new) == -1.:
                    R_new *= -1
                
                t_new = t_new / np.linalg.norm(t_new)
                
                self.P1 = self.K1 @ np.hstack((np.eye(3), np.zeros(3,1)))
                self.P2 = self.K2 @ np.hstack((R_new, t_new.reshape(-1, 1)))

                if self.check_cheirality():
                    return self.P1, self.P2

    def check_cheirality(self):
        X = Triangulate(self.P1, self.P2, self.pts1, self.pts2).manual(n=1)[0]
        X = np.hstack((X, 1))
        x1 = self.P1 @ X
        x2 = self.P2 @ X
        return x1[2] > 0 and x2[2] > 0

        
    