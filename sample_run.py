#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 13:21:57 2021

@author: soominkwon
"""

import numpy as np
import matplotlib.pyplot as plt
from lrpr_via_cgls import lrpr_fit

# importing sample data
data_name = 'mouse_small_data.npz'

with np.load(data_name) as sample_data:
    vec_X = sample_data['arr_0']
    Y = sample_data['arr_1']
    A = sample_data['arr_2']
    
# initializing parameters
image_dims = [10, 30]
rank = 1
iters = 5

# fitting new X
X_lrpr =  lrpr_fit(Y=Y, A=A, rank=rank, max_iters=iters)

X = np.reshape(vec_X, (image_dims[0], image_dims[1], -1), order='F')
X_lrpr = np.reshape(X_lrpr, (image_dims[0], image_dims[1], -1), order='F')

# plotting results
plt.imshow(np.abs(X[:, :, 0]), cmap='gray')
plt.title('True Image')
plt.show()

plt.imshow(np.abs(X_lrpr[:, :, 0]), cmap='gray')
plt.title('Reconstructed Image via LRPR')
plt.show()
    