import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import cv2

def cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

#inverse bilinear warp function: takes a set of warped, rasterized quads (a,b,c,d) and covered pixels (x) 
# and finds corresponding (s,t) coordinates that unwarp them
def inv_bilinear(x,a,b,c,d):
    e, f, g, hh = b - a, d - a, a - b + c - d, x - a
    k2, k1, k0 = cross(g, f), cross(e, f) + cross(hh, g), cross(hh, e)
    lin = np.abs(k2) < 1e-9
    k2s = np.where(lin, 1.0, k2)
    disc = np.sqrt(np.maximum(k1 * k1 - 4 * k0 * k2, 0))
    t1 = (-k1 - disc) / (2 * k2s)
    t2 = (-k1 + disc) / (2 * k2s)
    t_lin = -k0 / np.where(np.abs(k1) < 1e-12, 1e-12, k1)
    def s_of(t):  # pick the better-conditioned component
        den = e + g * t[..., None]
        num = hh - f * t[..., None]
        use_y = np.abs(den[..., 1]) > np.abs(den[..., 0])
        den_c = np.where(use_y, den[..., 1], den[..., 0])
        num_c = np.where(use_y, num[..., 1], num[..., 0])
        return num_c / np.where(np.abs(den_c) < 1e-12, 1e-12, den_c)
    s1, s2 = s_of(t1), s_of(t2)
    ok1 = (s1 >= -1e-3) & (s1 <= 1 + 1e-3) & (t1 >= -1e-3) & (t1 <= 1 + 1e-3)
    t = np.where(lin, t_lin, np.where(ok1, t1, t2))
    s = np.where(lin, s_of(t_lin), np.where(ok1, s1, s2))
    return np.clip(s, 0, 1), np.clip(t, 0, 1)

#convert [0,1] to pixel coordinates
def get_pixel_coords(pt,dims):
    pixel_coords = np.floor(pt*dims).astype(np.int32)
    pixel_coords[..., 0] = np.clip(pixel_coords[..., 0], 0, dims[0] - 1)
    pixel_coords[..., 1] = np.clip(pixel_coords[..., 1], 0, dims[1] - 1)
    return pixel_coords

#compute grid of anchor points (in coordnate-space)
def compute_grid(h,w, resolution=10):
    num_x = max(2, w // resolution)
    num_y = max(2, h // resolution)
    x = np.linspace(0, w-1, num_x)
    y = np.linspace(0, h-1, num_y)
    X, Y = np.meshgrid(x,y)
    grid = np.stack((X,Y), axis=-1)
    return grid

#get weight terms for each anchor point v for each control point p. Pre-process; run before render loop! These are 
#just constants. See mlm.py or ask Eero about what this means
def get_weights(P: np.ndarray, grid: np.ndarray, alpha=1.0, eps=1e-8):
    dist = np.sum((grid[:,:,None,:] - P[None, None, :, :])**2, axis=-1)
    w = 1.0 / (dist+eps)**alpha
    return w

def compute_constants_from_points(P: np.ndarray, grid: np.ndarray, mode="affine"):
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
    psum = wn @ P #shape (h,w,2)
    wsum = np.sum(wn, axis=-1, keepdims=True)
    pstar = psum / wsum #shape (h,w,2)
    #phat is the pstar-centered set of control points
    phat = P[None, None] - pstar[:, :, None, :]
    psum = wn @ P #shape (h,w,2)
    wsum = np.sum(wn, axis=-1, keepdims=True)
    #pstar is the centroid of control points relative to each v
    pstar = psum / wsum #shape (h,w,2)
    #phat is the pstar-centered set of control points
    phat = P[None, None] - pstar[:, :, None, :]
    vbar = grid - pstar #vbar is the pstar-centered set of grid points
    vbar_len = np.sqrt(np.einsum('hwx,hwx->hw', vbar, vbar)) #(h,w)
    if mode == "similarity":
        """PRECOMPUTE CONSTANTS FOR SIMILARITY TRANSFORMATIONS"""
        #orth((x,y)) = (-y,x)
        phat_orth = np.zeros_like(phat)
        phat_orth[:,:,:,0] = phat[:,:,:,1]
        phat_orth[:,:,:,1] = -phat[:,:,:,0]
        vbar_orth = np.zeros_like(vbar)
        vbar_orth[:,:,0] = vbar[:,:,1]
        vbar_orth[:,:,1] = -vbar[:,:,0]
        #mu_s = sum_i w_ip_ip_i^T ->sum of inner products between weighted control points and unweighted control points 
        mu_s = 1.0 / np.einsum('hwn,hwnx,hwnx->hw',w,phat,phat) #(h,w)
        pmat = np.stack([phat, phat_orth], axis=-2) #(h,w,n,2,2)
        vmat = np.stack([vbar, vbar_orth], axis=-2)
        A = np.einsum('hw,hwn,hwnxy,hwzy->hwnxz',mu_s,w,pmat,vmat) #paper equation 7. Output A is (h,w,n,2,2)
        return A, wn, vbar_len
    elif mode == "rigid":
        """PRECOMPUTE CONSTANTS FOR RIGID TRANSFORMATIONS"""
        phat_orth = np.zeros_like(phat)
        phat_orth[:,:,:,0] = phat[:,:,:,1]
        phat_orth[:,:,:,1] = -phat[:,:,:,0]
        vbar_orth = np.zeros_like(vbar)
        vbar_orth[:,:,0] = vbar[:,:,1]
        vbar_orth[:,:,1] = -vbar[:,:,0]
        pmat = np.stack([phat, phat_orth], axis=-2) #(h,w,n,2,2)
        vmat = np.stack([vbar, vbar_orth], axis=-2)
        A = np.einsum('hwn,hwnxy,hwzy->hwnxz',w,pmat,vmat) #paper equation 7, without similarity constant
        return A, wn, vbar_len
    """PRECOMPUTE CONSTANTS FOR AFFINE TRANSFORMATIONS"""
    #B is the outer product of all wi*phat and phat summed across all control points 
    B = np.einsum('hwn,hwnx,hwny->hwxy', w, phat, phat) #shape: (h, w, 2, 2)
    Binv = la.inv(B) 
    vbarBinv = np.einsum('hwx,hwxy->hwy',vbar,Binv) #EQ 5
    A = w * np.einsum('hwy,hwny->hwn', vbarBinv, phat) #EQ 5
    return A, wn, vbar_len
    

def get_affine(A: np.ndarray, wn: np.ndarray, q: np.ndarray):
    """
    computes f_a(v) = l_v for all v with AFFINE constraints
    A: (h, w, n) -> constant affine transformation scalars from P
    wn: (h,w,n) -> weights from P
    q: (n, 2) -> warped control point positions
    
    returns fa(v): (h,w,2) -> transformed grid vertices
    """
    qstar = np.einsum('hwn,nx->hwx', wn, q)
    qhat = q[None, None] - qstar[:, :, None, :]
    # shape (h,w,n) * (h,w,n,2) -> (h,w,n,2)
    qa = A[..., None] * qhat
    #shape (h, w, 2)
    transformed_v = np.sum(qa, axis=2) + qstar
    return transformed_v

def get_similarity(A: np.ndarray, wn: np.ndarray, q: np.ndarray):
    """
    computes f_a(v) = l_v for all v with SIMILARITY constraints
    A: (h, w, n, 2, 2) -> constant similarity transformation matrices from P
    wn: (h,w,n) -> weights from P
    q: (n, 2) -> warped control point positions

    returns fa(v): (h,w,2) -> transformed grid vertices
    """
    qstar = np.einsum('hwn,nx->hwx', wn, q)
    qhat = q[None, None] - qstar[:, :, None, :]
    return np.einsum('hwnx,hwnxy->hwy',qhat,A) + qstar

def get_rigid(A: np.ndarray, wn: np.ndarray, q: np.ndarray, vbar_len: np.ndarray):
    qstar = np.einsum('hwn,nx->hwx', wn, q)
    qhat = q[None, None] - qstar[:, :, None, :]
    f_r_v = np.einsum('hwnx,hwnxy->hwy',qhat,A)
    norm_fr = 1.0 / np.maximum(np.sqrt(np.einsum('hwx,hwx->hw', f_r_v, f_r_v)), 1e-12)
    return np.einsum('hw,hw,hwx->hwx', vbar_len, norm_fr, f_r_v) + qstar

