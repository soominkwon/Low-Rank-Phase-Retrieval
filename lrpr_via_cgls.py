#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 11:44:28 2021

@author: soominkwon
"""

import numpy as np
from custom_cgls_lrpr import cglsLRPR


def chooseRank(array):
    """
        Function to return the index of the largest difference between
        the j-th and (j+1)-th element.
    """
    
    array_len = array.shape[0]
    current_diff = 0
    idx = 0
    
    for i in range(1, array_len):
        diff = np.abs(array[i] - array[i-1])
    
        if diff > current_diff:
            current_diff = diff
            idx = i
            
    return idx


def lrprInit(Y, A, rank=None):
    """ Function to use spectral initialization for the factor matrices as described in
        Vaswani et al. (2017).
    
        Arguments:
            Y: Observation matrix with dimensions (m x q)
            A: Measurement tensor with dimensions (n x m x q)

    """    
    
    # initializing
    m = Y.shape[0]
    q = Y.shape[1]
    n = A.shape[0]

    Y_init = np.zeros((n, n), dtype=np.complex)
    
    # looping through each frame
    for k in range(q):
        y_k = Y[:, k]
        A_k = A[:, :, k]
        y_k_mean = y_k.mean()

        trunc_y_k = np.where(np.abs(y_k)<=9*y_k_mean, y_k, 0)
        Y_init += (A_k @ np.diag(trunc_y_k) @ A_k.T)
        
    Y_init = (1/(m*q))*Y_init
    
    eig_val, eig_vec = np.linalg.eig(Y_init)
    
    # choosing rank if not given
    if rank is None:
        rank = chooseRank(eig_val)
        
        # fixing the chosen rank if needed
        max_rank = min(n, q)
        if rank > max_rank:
            rank = max_rank
            
        U = eig_vec[:, :rank]
    else:
        U = eig_vec[:, :rank]
        
    B = np.zeros((q, rank), dtype=np.complex)
    
    for k in range(q):
        y_k = Y[:, k]
        A_k = A[:, :, k]
        mean_y_k = y_k.mean()
        
        avg_Y = (1/m)*(A_k @ np.diag(y_k) @ A_k.T)
        B_init_mat = U.T @ avg_Y @ U
        
        b_val, b_vec = np.linalg.eig(B_init_mat)
        b_k = np.sqrt(mean_y_k) * b_vec[:, 0]
        
        B[k] = b_k
     
    print('Chosen rank:', rank)
    print('Spectral Initialization Complete.')
    
    return U, B
    

def updateC(A, U, B):
    """ Function to update the diagonal phase matrix C.
    
        Arguments: 
            A: Measurement tensor with dimensions(n x m x q)
            U: Basis matrix with dimensions (n x r)
            B: Matrix with dimensions (q x r)
            
        Returns:
            C_tensor: Tensor where the frontal slices represent C_k (diagonal phase matrix)
                        with dimensions (m x m x q)
    """
    
    m_dim = A.shape[1]    
    q_dim = B.shape[0]
    
    C_tensor = np.zeros((m_dim, m_dim, q_dim), dtype=np.complex)
    
    for k in range(q_dim):
        A_k = A[:, :, k]
        b_k = B[k]
        
        x_hat = U @ b_k
        y_hat = A_k.T @ x_hat
        
        phase_y = np.exp(1j*np.angle(y_hat))
        #phase_y = np.sign(y_hat)
        C_k = np.diag(phase_y)
        C_tensor[:, :, k] = C_k
        
        
    return C_tensor


def lrpr_fit(Y, A, rank=None, max_iters=15):
    """
        Training loop for LRPR via CGLS.
    """

    # initializing factor matrices
    U, B = lrprInit(Y=Y, A=A, rank=rank)
    
    n, r = U.shape
    m = Y.shape[0]
    q = Y.shape[1]

    Y_sqrt = np.sqrt(Y)
    C_y_vec = np.zeros((m*q, ), dtype=np.complex);

    for i in range(max_iters):
        print('Current Iteration:', i)

        # update D
        st = 0
        en = m
    
        C_all = updateC(A=A, U=U, B=B)
    
        for k in range(q):
            C_y = C_all[:, :, k] @ Y_sqrt[:, k]
            C_y_vec[st:en] = C_y
            
            st += m
            en += m
        
        U_vec = cglsLRPR(A_sample=A, B_factor=B, C_y=C_y_vec)
        U = np.reshape(U_vec, (n, r), order='F')

        for k in range(q):
            A_k = A[:, :, k]
            y_k = Y_sqrt[:, k]
            C_k = C_all[:, :, k]
            
            M = A_k.T @ U
            
            # closed form for M
            b_k = np.linalg.inv(M.T @ M) @ M.T @ (C_k @ y_k)
            
            B[k] = b_k
            
    return U, B
