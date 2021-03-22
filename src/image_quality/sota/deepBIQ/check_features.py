import numpy as np
# from pickle import load
from joblib import load
# from sklearn.preprocessing._data import

# f1 = np.load(r'..\databases\AlexNet_features\train\koniq_normal\10004473376.npy')
# f2 = np.load(r'..\databases\AlexNet_features\train\koniq_small\10004473376.npy')
#
# f = np.array_equal(f1, f2)

# with open(r'..\databases\results\deepBIQ\scaler_all.obj', 'rb') as sf:
#     scaler = load(sf)
scaler = load(r'..\databases\results\deepBIQ\scaler_all_1.obj')
t = 0