from copy import deepcopy
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt

def k_means(data, k, c , m):

    clusters = np.zeros(m)
    distances = np.zeros((m,k))

    c_Old = np.zeros(c.shape)
    c_New = deepcopy(c)

    error = np.linalg.norm(c_New - c_Old)

    while error != 0:

        for i in range(k):
            distances[:,i] = np.linalg.norm(data - c[i], axis=1)
        clusters = np.argmin(distances, axis = 1)
        c_Old = deepcopy(c_New)
        
        for i in range(k):
            c_New[i] = np.mean(data[clusters == i], axis=0)
        error = np.linalg.norm(c_New - c_Old)

    return c_New

z = k_means(data, k, centers , n)