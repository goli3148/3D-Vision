from sklearn.cluster import DBSCAN
import numpy as np
class ROL:
    def __call__(self, point_clouds, colors) -> None:
        db = DBSCAN(eps=.3, min_samples=10).fit(point_clouds)
        labels = db.labels_

        # Labels with -1 are outliers
        outliers = point_clouds[labels == -1]
        inliers = point_clouds[labels != -1]

        return inliers, colors[labels != -1]

