import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import cv2
from utils import compute_constants_from_centered, tangent_sum

#hyperbolic math helpers
def c2v(z):  return np.stack([z.real, z.imag], -1) #convert complex coord to 2-vector
def v2c(v):  return v[..., 0] + 1j * v[..., 1] #convert 2-vector to single complex coord
def tau(a, z): return (z - a) / (1 - np.conj(a) * z)
def tau_inv(a, z): return (z + a) / (1 + np.conj(a) * z)
def log0(z):
    r = np.abs(z); return np.where(r > 1e-15, 2 * np.arctanh(r) / np.maximum(r, 1e-15), 2.0) * z
def exp0(v):
    r = np.abs(v); return np.where(r > 1e-15, np.tanh(r / 2) / np.maximum(r, 1e-15), 0.5) * v

#coordinate mapping helpers

def image_to_disk(x, image_center, K, half_diag):
    """Map image (Euclidean Plane) to poincare disk (projection of Hyperbolic Plane)
    Args:
        x: image coordinates 
        image_center: coordinates of center
        K: hyperbolic scaling factor. One image half-diagonal is equal to K hyperbolic units.
        half_diag: length of half the image's diagonal
    Returns:
        poincare disk coordinates of input image points
    """
    d = x - image_center #compute centered coordinates
    r = np.hypot(d[..., 0], d[..., 1]) #get distance from center
    rho = K * r / half_diag #convert distance to hyperbolic units
    return np.tanh(rho / 2) * np.exp(1j * np.arctan2(d[..., 1], d[..., 0])) #get poincare disk position z=x+iy

def disk_to_image(z, image_center, K, half_diag):
    r = np.abs(z); rho = 2 * np.arctanh(np.minimum(r, 1 - 1e-12))
    s = np.where(r > 1e-15, rho / np.maximum(r, 1e-15), 2.0) * half_diag / K
    return image_center + np.stack([z.real, z.imag], -1) * s[..., None]

def disk_to_screen(z, c, r_px):
    """Convert poincare disk coordinates z = x+iy to screen coordinates
    Args:
        z: complex poincare disk coordinates
        c: screen center
        r: canvas radius in pixels
    Returns:
        screen coordinates of input z
    """
    return np.stack((z.real, z.imag),axis=-1)*r_px + c

def screen_to_disk(s, c, r_px,max_r=0.995):
    """Convert screen coordinates to a poincare disk coordinate, z
    Args:
        s: screen coordinates of input points
        c: screen center
        r_px: canvas radius in pixels
        max_r: maxmimum magnitude of poincare coordinate still considered 'on the disk'
    Returns:
        poincare disk coordinates of input screen coordnates
    """
    d = (s - c)/r_px
    z =d[...,0]+1j*d[...,1]
    r = np.abs(z)
    return np.where(r>max_r, z * (max_r / np.maximum(r, 1e-12)), z)

#precompute helpers
def lorentz_centroid(wn, z_pts):
    r2 = np.abs(z_pts) ** 2
    X = np.stack([1 + r2, 2 * z_pts.real, 2 * z_pts.imag], -1) / (1 - r2)[:, None] #lift onto sheet
    s = np.einsum('hwn,nk->hwk', wn, X) # average
    s /= np.sqrt(s[..., 0]**2 - s[..., 1]**2 - s[..., 2]**2)[..., None] # back onto sheet
    return (s[..., 1] + 1j * s[..., 2]) / (1 + s[..., 0]) #centroid project back to z

def compute_constants_from_hyperbolic_points(z_p, z_grid, w, wn, mode="rigid"):
    ps = lorentz_centroid(wn, z_p)
    phat = c2v(log0(tau(ps[..., None], z_p[None, None])))
    vhat = c2v(log0(tau(ps, z_grid)))
    A, vbl = compute_constants_from_centered(phat, vhat, w, mode)
    return A, vbl

#deformation function
def deform_hyperbolic(A, wn, z_q, mode, vbl):
    qs = lorentz_centroid(wn, z_q)
    qhat = c2v(log0(tau(qs[...,None],z_q[None,None])))
    return tau_inv(qs, exp0(v2c(tangent_sum(A,qhat,mode,vbl))))