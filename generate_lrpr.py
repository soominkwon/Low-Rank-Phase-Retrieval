#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:03:27 2021

@author: soominkwon
"""


import numpy as np

def generateLRPRMeasurements(image_name, m_dim):
    """ Function to obtain measurements y's (m x q) and A's (m x n x q).
    
        Arguments:
            image_name: name of .npz file to load (n1 x n2 x q)
            m_dim: dimensions of m
    
    """
    
    with np.load(image_name) as data:
        tensor = data['arr_0']
        
    q_dim = tensor.shape[2]
    vec_images = np.reshape(tensor, (-1, q_dim), order='F')
    
    n_dim = vec_images.shape[0]
    
    A_tensor = np.zeros((n_dim, m_dim, q_dim))
    Y = np.zeros((m_dim, q_dim))
    
    for k in range(q_dim):
        A_k = np.random.randn(n_dim, m_dim)
        A_tensor[:, :, k] = A_k
        x_k = vec_images[:, k]
        
        norm_x_k = np.linalg.norm(x_k)
        x_k = x_k / norm_x_k
        
        y_k = np.abs(A_k.T @ x_k)**2
        Y[:, k] = y_k
    
    return tensor, Y, A_tensor
