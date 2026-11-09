import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse as ap
import cv2
from pathlib import Path
import sys
import math

from utils import get_weights, get_affine, compute_constants_from_points

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

def place_points(img, points):
    selected = None
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
        if key == 27 or key == ord("q"):  #Esc or Q to quit
            break
        elif key == ord("r"):  #R to reset
            points.clear()
            state["selected"] = None

    cv2.destroyAllWindows()

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
    
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    points = []
    place_points(img, points)


if __name__ == "__main__":
    main()