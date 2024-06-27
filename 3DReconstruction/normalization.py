import numpy as np

class Normalization:
    def __init__(self, pts1, pts2):
        self.pts1 = pts1
        self.pts2 = pts2

    def normalize(self):
        self.pts1 = self.pts1.reshape(-1, 2)
        self.pts2 = self.pts2.reshape(-1, 2)

        mat1 = self.norm_matrix(np.mean(self.pts1, 0), np.std(self.pts1))
        mat2 = self.norm_matrix(np.mean(self.pts2, 0), np.std(self.pts2))

        self.pts1 = (np.hstack((self.pts1, np.ones((self.pts1.shape[0], 1)))) @ mat1).T
        self.pts2 = (np.hstack((self.pts2, np.ones((self.pts2.shape[0], 1)))) @ mat1).T
        self.pts1, self.pts2 = self.pts1/self.pts1[2], self.pts2/self.pts2[2]


        return self.pts1[:2].T.reshape(-1, 1, 2), self.pts2[:2].T.reshape(-1, 1, 2)

    def norm_matrix(self, mean, std):
        mat = np.array([
            [std/np.sqrt(2),   0,          mean[0]],
            [0,         std/np.sqrt(2),    mean[1]],
            [0,         0,          1   ]
        ])
        return np.linalg.inv(mat)