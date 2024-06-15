import numpy as np
from scipy.linalg import null_space
class P2M:
    def __init__(self, P) -> None:
        self.P = P

    def decompositionMethod(self):
        M = self.P[:, 0:3]
        
        M = np.linalg.inv(M)
        
        Q, K = np.linalg.qr(M)
        
        K = np.linalg.inv(K)
        K /= K[2, 2]
        return K
    
    # def decompositionMethod(self):
    #     C = null_space(self.P1)
    #     # Normalize the homogeneous coordinate
    #     C /= C[-1] 
    #     # Compute the skew-symmetric matrix of P2 * C
    #     first = (self.P2 @ C).reshape(3,)
    #     skew_first = np.array([
    #         [0, -first[2], first[1]],
    #         [first[2], 0, -first[0]],
    #         [-first[1], first[0], 0]
    #     ])
    #     # Compute P2 * pseudo-inverse of P1
    #     second = self.P2 @ np.linalg.pinv(self.P1)
    #     # Compute the fundamental matrix
    #     self.F = skew_first @ second
    #     return self.F
    

    # def compute_pseudo_inverse(self, P):
    #     """Compute the Moore-Penrose pseudo-inverse of the camera matrix P."""
    #     return np.linalg.pinv(P)

    # def compute_epipole(self, P_prime):
    #     """Compute the epipole in the second view from the camera matrix P'."""
    #     # The epipole is the null space of P'
    #     U, S, Vt = np.linalg.svd(P_prime)
    #     e_prime = Vt[-1]
    #     e_prime /= e_prime[-1]  # Normalize to make the last entry 1
    #     return e_prime

    # def skew_symmetric_matrix(self, v):
    #     """Construct a skew-symmetric matrix from a vector v."""
    #     return np.array([
    #         [0, -v[2], v[1]],
    #         [v[2], 0, -v[0]],
    #         [-v[1], v[0], 0]
    #     ])

    # def epipoleMethod(self):
    #     """Compute the fundamental matrix F from camera matrices P and P'."""
    #     P = self.P1
    #     P_prime = self.P2
    #     P_pseudo_inverse = self.compute_pseudo_inverse(P)
    #     e_prime = self.compute_epipole(P_prime)
    #     skew_symmetric = self.skew_symmetric_matrix(e_prime)
    #     P_prime_P_pseudo_inverse = np.dot(P_prime, P_pseudo_inverse)
    #     self.F = np.dot(skew_symmetric, P_prime_P_pseudo_inverse)
    #     return self.F
