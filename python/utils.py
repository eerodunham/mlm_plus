import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la

def get_weights(P: np.ndarray, grid: np.ndarray, alpha=1, eps=1e-8):
    dist = ((grid[:,:,None,:] - P[None, None, :, :])**2)
    w = 1.0 / (dist+eps)**alpha
    return w

def compute_constants_from_points(P: np.ndarray, grid: np.ndarray):
    """Compute constants only dependant on control points P.

    Args:
        P: numpy array of shape ``(Npoints, 2)`` containing the floating point 
            locations of each new control point the user has placed
        grid: numpy array of shape ``(grid height, grid width, 2)`` containing
            the floating point vertex locations of the deformation grid

    Returns:
        numpy array of shape ``(grid height, grid width, n_scalars)`` containing each
        grid point ``v``'s constant matrix A_v

    """
    #compute weights for each control point. grid shape: (h,w,2); points shape: (npts, 2); 
    #output shape: (h,w,n)
    w = get_weights(P, grid)
    wn = w / w.sum(axis=-1, keepdims=True)
    #pstar is the weighted sum of all control points distance from each grid point v, with shape (h,w,2)
    #first, get weighted sum of each control point using matrix multiplication
        #common dim is n; for each vector i,j in the output, this computes np.dot(weights_col_j, points_row_i) for both elements
    psum = wn @ P
    #then divide by sums of weights. phat has shape (h,w,2)
    pstar = psum / np.sum(wn, axis=-1, keepdims=True)
    #phat is each control point centered at each grid point; shape: (h,w,n,2)
    phat = P[None, None] - pstar[:, :, None, :]
    #compute B; which is the outer product of all wi*phat and phat summed across all control points\
    B = np.einsum('hwi,hwix,hwiy->hwxy', w, phat, phat)
    #shape: (h, w, 2, 2)
    Binv = la.inv(B)
    vbar = grid - pstar
    vbarBinv = np.einsum('hwx,hwxy->hwy',vbar,Binv)
    A = w * np.einsum('hwx,hwny->hwn', vbarBinv, phat)
    return A, wn

def get_affine(A: np.ndarray, q: np.ndarray, qstar: np.ndarray):
    """
    computes f_a(v) = l_v for all v with AFFINE constraints
    A: (h, w, n) -> constant scalars from P
    q: (n, 2) -> warped control point positions
    qstar: (h,w,2) -> mean warped control point relative to each grid vertex

    returns fa(v): (h,w,2) -> transformed grid vertices
    """
    #shape (n,2) - (h,w,2) -> (h,w,n,2)
    qhat = q[None, None] - qstar[:, :, None, :]
    #shape (h,w,n) * (h,w,n,2) -> (h,w,n,2)
    qa = A[..., None] * qhat
    #shape (h, w, 2)
    transformed_v = np.sum(qa, axis=2) + qstar
    return transformed_v