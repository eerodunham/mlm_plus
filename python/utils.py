import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import cv2

def get_weights(P: np.ndarray, grid: np.ndarray, alpha=1.0, eps=1e-8):
    dist = np.sum((grid[:,:,None,:] - P[None, None, :, :])**2, axis=-1)
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
    w = get_weights(P, grid) #shape: (h,w,n)
    wn = w / w.sum(axis=-1, keepdims=True) 
    #pstar is the centroid of control points relative to each v
    psum = wn @ P #shape (h,w,2)
    wsum = np.sum(wn, axis=-1, keepdims=True)
    pstar = psum / wsum #shape (h,w,2)
    #phat is the pstar-centered set of control points
    phat = P[None, None] - pstar[:, :, None, :]
    #B is the outer product of all wi*phat and phat summed across all control points 
    B = np.einsum('hwn,hwnx,hwny->hwxy', w, phat, phat) #shape: (h, w, 2, 2)
    Binv = la.inv(B) 
    vbar = grid - pstar #vbar is the pstar-centered set of grid points
    vbarBinv = np.einsum('hwx,hwxy->hwy',vbar,Binv) #EQ 5
    A = w * np.einsum('hwy,hwny->hwn', vbarBinv, phat) #EQ 5
    return A, wn

def get_affine(A: np.ndarray, wn: np.ndarray, q: np.ndarray):
    """
    computes f_a(v) = l_v for all v with AFFINE constraints
    A: (h, w, n) -> constant scalars from P
    q: (n, 2) -> warped control point positions
    qstar: (h,w,2) -> mean warped control point relative to each grid vertex

    returns fa(v): (h,w,2) -> transformed grid vertices
    """
    qstar = np.einsum('hwn,nx->hwx', wn, q)
    qhat = q[None, None] - qstar[:, :, None, :]
    # shape (h,w,n) * (h,w,n,2) -> (h,w,n,2)
    qa = A[..., None] * qhat
    #shape (h, w, 2)
    transformed_v = np.sum(qa, axis=2) + qstar
    return transformed_v
