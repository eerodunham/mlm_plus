import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse as ap
import cv2
from pathlib import Path
import sys
import math

from utils import get_weights, get_affine, compute_constants_from_points, compute_grid, get_pixel_coords, inv_bilinear

RAD = 5
DRAG_RAD = RAD*2

def click_event_place(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(param["points"]):
            dist = math.hypot(x-pt[0], y-pt[1])
            if dist <= DRAG_RAD:
                param["selected"] = i
                break

        if param["selected"] is None:
            param["points"].append([x,y])

    elif event == cv2.EVENT_MOUSEMOVE:
         if param["selected"] is not None:
            param["points"][param["selected"]] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        param["selected"] = None

def click_event_warp(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(param["points"]):
            dist = math.hypot(x-pt[0], y-pt[1])
            if dist <= DRAG_RAD:
                param["selected"] = i
                break

    elif event==cv2.EVENT_MOUSEMOVE:  
        if param["selected"] is not None:
            param["points"][param["selected"]] = (x,y)

    elif event == cv2.EVENT_LBUTTONUP:
            param["selected"] = None

def place_points(img, points):
    print("-----++++++INSTRUCTIONS++++++-----\nClick to place a new control point. Click and drag to move an existing one. Q or ESC to exit window. R to reset points.")
    state = {"points": points, "image": img, "selected": None}
    cv2.namedWindow("Place Control Points")
    cv2.setMouseCallback("Place Control Points", click_event_place, param=state)

    while True:
        disp_img = img.copy()

        for i, pt in enumerate(points):
            color = (0, 0, 255) if i == state["selected"] else (0, 255, 0)
            cv2.circle(disp_img, pt, RAD, color, -1)

        cv2.imshow("Place Control Points", disp_img)

        key = cv2.waitKey(1) & 0xFF
        #ESCAPE or Q to quit
        if key == 27 or key == ord("q"):
            break
        # R to reset
        elif key == ord("r"):
            points.clear()
            state["selected"] = None

    cv2.destroyAllWindows()

def draw_grid(img, grid, dims):
    cv2.namedWindow("Grid computed")
    grid_img = img.copy()
    h, w, _ = grid.shape
    grid_int = get_pixel_coords(grid, dims)
    for i in range(h):
        for j in range(w):
            pt = grid_int[i,j]
            cv2.circle(grid_img, pt, 1, (255,0,0), -1)
    cv2.imshow('Grid computed', grid_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def render_warp(img, g, gt, shift=4):
    h, w = img.shape[:2]
    gh, gw = g.shape[:2]
    raster_ids = np.full((h,w),-1,np.int32)
    #multiply by 2^shift
    fixed = np.round(gt * (1 << shift)).astype(np.int32)
    for r in range(gh-1):
        for c in range(gw-1):
            quad = fixed[[r,r,r+1,r+1], [c,c+1,c+1,c]]
            #rasterize deformed quad
            cv2.fillPoly(raster_ids, [quad], int(r * (gw - 1) + c), lineType=cv2.LINE_8, shift=shift)
    #get coordinate pairs, rows and columns of all covered pixels
    ys, xs = np.nonzero(raster_ids >= 0)
    x = np.stack([xs, ys], -1).astype(np.float32)
    k = raster_ids[ys,xs]
    r = k // (gw-1)
    c = k % (gw-1)
    A = gt[r,c]
    B = gt[r,c+1]
    C = gt[r+1,c+1]
    D = gt[r+1,c]
    s, t = inv_bilinear(x,A,B,C,D)
    A0, B0, C0, D0 = g[r, c], g[r, c + 1], g[r + 1, c + 1], g[r + 1, c]
    src = ((1 - s) * (1 - t))[:, None] * A0 + (s * (1 - t))[:, None] * B0 \
        + (s * t)[:, None] * C0 + ((1 - s) * t)[:, None] * D0
    map_x = np.full((h, w), -1, np.float32); map_y = np.full((h, w), -1, np.float32)
    map_x[ys, xs], map_y[ys, xs] = src[:, 0], src[:, 1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

def move_points_affine(img, A, wn, q):
    window_name = "Affine Deformation with MLM"
    cv2.namedWindow(window_name)
    state = {"points": q, "selected": None}
    cv2.setMouseCallback(window_name, click_event_warp, param=state)
    h, w, _ = img.shape
    grid = compute_grid(h,w,10)
    warped = img.copy()
    last_q = q.copy()
    
    while True:
        if not np.array_equal(q, last_q):
            grid_transformed = get_affine(A, wn, q)
            warped = render_warp(img, grid, grid_transformed)
            last_q = q.copy()

        disp = warped.copy()

        for i, pt in enumerate(q):
            color = (0, 0, 255) if i == state["selected"] else (0, 255, 0)
            cv2.circle(disp, tuple(int(c) for c in pt), RAD, color, -1)

        cv2.imshow(window_name, disp)

        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break

#MAIN FUNCTION
def main():
    parser = ap.ArgumentParser(description="MLM Python Demo")
    parser.add_argument("path", type=str, help="Path to image file.")
    args = parser.parse_args()

    img_path = Path(args.path)
    if img_path.exists():
        print(f'Opening image at path {args.path}')
    else:
        raise FileNotFoundError("Image file not found!")

    img = cv2.imread(args.path)
    if img is None:
        raise ValueError("Image could not be read.")
    h, w, _ = img.shape
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points = []
    place_points(img, points)
    #compute constants from control points
    if(len(points) == 0):
        print("adding center point")
        points.append((w//2, h//2))
    P = np.array(points).astype(np.float32)
    q = P.copy()
    grid = compute_grid(h,w)
    # draw_grid(img, grid, dims)
    A, wn = compute_constants_from_points(P, grid)
    move_points_affine(img, A, wn, q)


if __name__ == "__main__":
    main()