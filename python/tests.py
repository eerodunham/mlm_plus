import numpy as np

from utils import compute_constants_from_points, get_affine, get_weights

def test_affine():
    rng = np.random.default_rng(0)
    P = rng.uniform(0, 10, (6, 2))
    ys, xs = np.mgrid[0:10:40j, 0:10:50j]
    grid = np.stack([xs, ys], -1)
    A, wn = compute_constants_from_points(P, grid)
    print("identity err:", np.abs(get_affine(A, wn, P) - grid).max())
    L = np.array([[1.3, 0.2], [-0.4, 0.8]]); t = np.array([2., -1.])
    print("global affine err:", np.abs(get_affine(A, wn, P @ L + t) - (grid @ L + t)).max())
    q = P + rng.normal(0, 1, P.shape)
    Ap, wnp = compute_constants_from_points(P, P[None])  # evaluate at control points
    print("interpolation err:", np.abs(get_affine(Ap, wnp, q)[0] - q).max())

def main():
    print("AFFINE WARP TEST")
    test_affine()

if __name__ == "__main__":
    main()