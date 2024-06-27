import numpy as np

class ScaleAndAlignment:
    def __init__(self) -> None:
        pass

    def center_point_cloud(self, points):
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        return centered_points, centroid

    def estimate_scale_factor(self, points1, points2):
        # Flatten the arrays to compute the scale factor using least squares
        A = np.vstack([points1.flatten(), np.ones(points1.size)]).T
        b = points2.flatten()
        scale, _ = np.linalg.lstsq(A, b, rcond=None)[0]
        return scale

    def align_point_clouds(self, points1, points2):
        # Step 1: Center both point clouds
        centered_points1, centroid1 = self.center_point_cloud(points1)
        centered_points2, centroid2 = self.center_point_cloud(points2)
        
        # Step 2: Estimate the scale factor
        scale_factor = self.estimate_scale_factor(centered_points2, centered_points1)
        
        # Step 3: Scale the second point cloud
        scaled_points2 = centered_points2 * scale_factor
        
        # Step 4: Translate the scaled point cloud back to original position
        scaled_points2 += centroid1
        
        return centered_points1 + centroid1, scaled_points2