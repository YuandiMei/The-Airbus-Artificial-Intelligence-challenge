import pandas as pd
import math
import numpy as np

# Read the origin data from the .h5 file
def get_raw_data(path, filename):
    return pd.read_hdf(path+'/'+filename).values

# Tranform from the raw data into signals
def file_to_signals(path, filename, flatten=False):
    x_1 = pd.read_hdf(path+'/'+filename)
    x = x_1.values
    y = []
    for i in range(x.shape[0]):
        y.append(x[i,:])
    Y = np.asarray(y)
    return Y
