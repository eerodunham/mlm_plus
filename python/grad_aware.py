import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.ndimage import map_coordinates

import cv2

#precompute distances based on minimizing the energy function equal to sum(|grad|)
def d2_geodesic(P, grid, img, step=2, blur_sigma=1.5, B=25, eps=1e-9):
    cost = compute_cost(img, step, blur_sigma, B)
    gh, gw = cost.shape
    n_nodes = gh * gw
    n_points = len(P)
    r, c, v = gradient_graph(cost, step)
    p_sub= P/step
    sr, sc, sv = [],[],[]
    for i, (x,y) in enumerate(p_sub):
        x0, y0 = int(np.clip(np.floor(x), 0, gw - 2)), int(np.clip(np.floor(y), 0, gh - 2))
        for yy, xx in [(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)]:
            sr.append(n_nodes + i)
            sc.append(yy * gw + xx)
            sv.append(max(step * np.hypot(x - xx, y - yy) * cost[yy, xx], eps))
    G = coo_matrix((np.r_[v, sv], (np.r_[r, sr], np.r_[c, sc])), shape=(n_nodes + n_points, n_nodes + n_points)).tocsr()
    D = dijkstra(G, directed=False, indices=np.arange(n_nodes, n_nodes + n_points)) #shape (n_points, n_nodes+n_points)
    dist_graph = D[:,:n_nodes].reshape(n_points, gh, gw) #we only care about distances to each grid point
    coords = [grid[..., 1].ravel() / step, grid[..., 0].ravel() / step] #take every step grid vertices
    Dg = np.stack([map_coordinates(Di, coords, order=1, mode='nearest') for Di in dist_graph], -1) #map downsampled distances to grid vertices
    return Dg.reshape(grid.shape[0], grid.shape[1], n_points) ** 2 #reshape to (h, w, n), square and return

#compute the gradient magnitude across the image using a sobel filter
def compute_cost(img, step, blur_sigma, B):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255
    gray = cv2.GaussianBlur(gray, (0,0), blur_sigma) #blur to smooth features so small gradient changes don't affect image
    #IMPROVEMENT: blur more, take laplacian, etc
    gradient = np.hypot(cv2.Sobel(gray, cv2.CV_64F, 1, 0), cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    gradient /= np.percentile(gradient, 99) + 1e-12 #Robustness: normalize with respect to non-outlier max
    max_pool = cv2.dilate(gradient, np.ones((step,step))) #take max value in subsampling region with uniform weights
    cost = 1 + B* np.clip(max_pool[::step, ::step],0,1) #sub-sample by step in both directions. clip and rescale.
    return cost #cost is 1+B*|Del|; shape (gh,gw)

#build and return rows, columns and edge weights  
def gradient_graph(cost_map, step):
    gh, gw = cost_map.shape
    idx = np.arange(gw*gh).reshape(gh,gw)
    rows, cols, vals = [],[],[]
    gradients = [(0,1),(1,0),(1,1),(1,-1)]
    for dy, dx in gradients:
        a = idx[0:(gh-dy), max(0, -dx):(gw - max(0, dx))]
        b = idx[dy:, max(0, dx):(gw + min(0, dx))]
        ca, cb = cost_map.ravel()[a], cost_map.ravel()[b]
        rows.append(a.ravel())
        cols.append(b.ravel())
        vals.append((step * np.hypot(dx, dy) * 0.5 * (ca + cb)).ravel())
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)
