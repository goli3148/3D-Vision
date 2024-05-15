import numpy as np
import matplotlib.pyplot as plt

def data():
    # Define the covariance matrices for the circular data and line data
    cov_circle = np.array([[.5, 0], [0, 0.5]])
    cov_line = np.array([[3, 4], [4, 6]])

    # Generate random data points for the circular data
    num_samples_circle = 30
    circle_data = np.random.multivariate_normal([-3, 4], cov_circle, num_samples_circle)
    circle_data2 = np.random.multivariate_normal([4, -3], cov_circle, num_samples_circle)
    # Generate random data points for the line data
    num_samples_line = 100
    line_data = np.random.multivariate_normal([2, 2], cov_line, num_samples_line)
    data_set = np.vstack((circle_data, line_data))
    data_setnew=np.vstack((data_set, circle_data2))
    # Plot the data sets
    # plt.figure(figsize=(8, 8))
    # plt.scatter(data_setnew[:, 0], data_setnew[:, 1], c='r', marker='x', label='My dataset')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Generated Data Set')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    return data_setnew