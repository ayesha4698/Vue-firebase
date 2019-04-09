import numpy as np
import math
import cv2

import scipy
from scipy.signal import convolve2d as convolve
import matplotlib.pyplot as plt

filter = np.array([[1, 2, 1],
                   [2, 4, 2],
                   [1, 2, 1]])
filter = np.multiply( filter, (1/np.sum(filter)) )

image = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

result = convolve(image, filter, boundary='symm', mode='same')



print(result);


a = np.array([1, 2, 1])
b = np.array([1, 2, 1])
c = np.outer(a, b)
print(c)


