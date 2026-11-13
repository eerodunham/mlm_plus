import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse as ap
import cv2
from pathlib import Path
import sys
import math
from typing import Any #to silence linter errors
from grad_aware import d2_geodesic

from utils import deform_euclidean, compute_constants_from_points, compute_grid, precompute_weights, inv_bilinear
from hyperbolic import deform_hyperbolic, screen_to_disk, disk_to_screen, image_to_disk, disk_to_image, compute_constants_from_hyperbolic_points
WARPS   = ["affine", "similarity", "rigid"]
METRICS = ["euclidean", "polar", "gradient", "hyperbolic"]

RAD = 2
DRAG_RAD = RAD*2
PX = 250

RIPPLE_K_DISK = 0.3

MIN_HYP_R = 1.10
MAX_HYP_R = 2

MIN_LAM = 0.05
MAX_LAM = 5

def slider_event(x):
    pass

def click_event_place(event, x, y, flags, param):
    if event==cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(param["points"]):
            dist = math.hypot(x-pt[0], y-pt[1])
            if dist <= param["drag radius"]:
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
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(param["to_screen"](param["points"])):
            if math.hypot(x - pt[0], y - pt[1]) <= param["drag radius"]:
                param["selected"] = i; break
    elif event == cv2.EVENT_MOUSEMOVE and param["selected"] is not None:
        param["points"][param["selected"]] = param["from_screen"](np.array([x, y], float))
    elif event == cv2.EVENT_LBUTTONUP:
        param["selected"] = None

# def click_event_warp(event, x, y, flags, param):
#     if event==cv2.EVENT_LBUTTONDOWN:
#         for i, pt in enumerate(param["points"]):
#             dist = math.hypot(x-pt[0], y-pt[1])
#             if dist <= param["drag radius"]:
#                 param["selected"] = i
#                 break

#     elif event==cv2.EVENT_MOUSEMOVE:  
#         if param["selected"] is not None:
#             param["points"][param["selected"]] = (x,y)

#     elif event == cv2.EVENT_LBUTTONUP:
#             param["selected"] = None

def place_points(img, points):
    print("-----++++++INSTRUCTIONS++++++-----\nClick to place a new control point. Click and drag to move an existing one. Q or ESC to exit window. R to reset points.")
    h,w,_ = img.shape
    scale = max(h,w) / PX
    radius = np.floor(RAD*scale).astype(int)
    drag_radius = np.floor(DRAG_RAD*scale)
    state = {"points": points, "image": img, "selected": None, "drag radius": drag_radius}
    cv2.namedWindow("Place Control Points")
    cv2.setMouseCallback("Place Control Points", click_event_place, param=state)
    
    while True:
        disp_img = img.copy()

        for i, pt in enumerate(points):
            color = (0, 0, 255) if i == state["selected"] else (0, 255, 0)
            cv2.circle(disp_img, pt, radius, color, -1)

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


def render_warp(img, g, gt, out_dims, shift=4):
    h, w = out_dims
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

def move_points(img,p,name,K=2.5,S=800,grid_res=10):
    h, w = img.shape[:2]
    scale = max(h,w) / PX
    win = "Image Deformation with MLS"
    cv2.namedWindow(win)
    grid = compute_grid(h, w, grid_res)
    MARGIN = 40
    S = max(S, max(h, w) + 2 * MARGIN)                  # one canvas size for every mode
    scr_c = np.array([(S - 1) / 2] * 2)
    r_px = S / 2 - 10
    offset = np.array([(S - w) / 2, (S - h) / 2])  
    img_c = np.array([(w - 1) / 2, (h - 1) / 2])
    half_diag = np.hypot(*img_c)
    r_px = S / 2 - 10
    to_disk = lambda x: image_to_disk(x, img_c, K, half_diag) #convert image coords x to disk coords z
    from_disk = lambda z: disk_to_image(z, img_c, K, half_diag) #convert poincare disk coords z to image coords x
    z_grid = to_disk(grid) 
    z_p = to_disk(p)
    geo_d2 = d2_geodesic(p, grid, img)
    q = p.copy() #image space coords
    warp_state = dict(metric=0, warp=0, ripple=False, lam=MIN_LAM * half_diag)
    state = {"points": q, "selected": None, "drag radius": DRAG_RAD,
             "to_screen": None, "from_screen": None}
    cv2.setMouseCallback(win, click_event_warp, param=state)
    cv2.createTrackbar("Lambda", win, 0, 100, lambda _: None)
    hyp = lambda: METRICS[warp_state["metric"]] == "hyperbolic"
    def rebuild():
        metric = METRICS[warp_state["metric"]]
        warp = WARPS[warp_state["warp"]]
        if hyp():
            wt, wn = precompute_weights(z_p, z_grid, "disk", eps=1e-12, ripple=bool(warp_state["ripple"]), K=RIPPLE_K_DISK)
            A, vbl = compute_constants_from_hyperbolic_points(z_p, z_grid, wt, wn, mode=warp)
            state["to_screen"]   = lambda x: disk_to_screen(to_disk(x), scr_c, r_px)
            state["from_screen"] = lambda y: from_disk(screen_to_disk(y, scr_c, r_px))
        else:
            dist_kw = {"polar": dict(center=img_c, lam=warp_state["lam"]), "gradient": dict(dist=geo_d2)}.get(metric, {})
            wt, wn = precompute_weights(p, grid, metric, ripple=bool(warp_state["ripple"]), dist_kw=dist_kw)
            A, vbl = compute_constants_from_points(p, grid, wt, wn, mode=warp)
            state["to_screen"] = lambda x: x+offset
            state["from_screen"] = lambda y: y - offset
        return A, wn, vbl
    
    def render(A, wn, vbl):
        warp = WARPS[warp_state["warp"]]

        if hyp():
            dest = disk_to_screen(deform_hyperbolic(A, wn, to_disk(q), warp, vbl), scr_c, r_px)
            out = render_warp(img, grid, dest, (S, S))
            return out
        
        return render_warp(img, grid, deform_euclidean(A, wn, q, warp, vbl)+offset, (S, S))

    def load_image(new_img, new_p):
        nonlocal img, p, h, w, grid, img_c, half_diag, offset, z_grid, z_p, geo_d2, model, last_q
        img = new_img
        h, w = img.shape[:2]
        grid = compute_grid(h, w, grid_res)
        img_c = np.array([(w - 1) / 2, (h - 1) / 2])
        half_diag = np.hypot(*img_c)
        offset = np.array([(S - w) / 2, (S - h) / 2])
        p = np.asarray(new_p, float)
        q[:] = p 
        z_grid, z_p = to_disk(grid), to_disk(p)
        geo_d2 = d2_geodesic(p, grid, img)
        model, last_q = rebuild(), None  

    def bake():
        shown = state["to_screen"](q) # handle positions on the canvas
        x, y, bw, bh = cv2.boundingRect(warped.any(-1).astype(np.uint8))# crop to bounding box around the content
        load_image(warped[y:y + bh, x:x + bw].copy(), shown - [x, y])

    model, dirty, last_q = rebuild(), False, None
    # MAIN RENDER LOOP
    while True:
        t = cv2.getTrackbarPos("Lambda", win) / 100
        lam = ((1 - t) * MIN_LAM + t * MAX_LAM) * half_diag
        if lam != warp_state["lam"]:
            warp_state["lam"] = lam
            dirty |= METRICS[warp_state["metric"]] == "polar"
        if dirty:
            model, dirty, last_q = rebuild(), False, None
        if last_q is None or not np.array_equal(q, last_q):
            warped = render(*model)
            last_q = q.copy()
        disp = warped.copy()
        for i, pt in enumerate(state["to_screen"](q)):
            color = (0, 0, 255) if i == state["selected"] else (0, 255, 0)
            cv2.circle(disp, tuple(int(c) for c in pt), int(scale*RAD), color, -1)

        if hyp():
            cv2.circle(disp, tuple(int(c) for c in scr_c), int(r_px), (90, 90, 90), 1) #render outer rim

        cv2.setWindowTitle(win, f"{WARPS[warp_state['warp']]} | {METRICS[warp_state['metric']]} | ripple "
                                f"{'on' if warp_state['ripple'] else 'off'}   [m]ode [d]istance [r]ipple [s]ave [q]uit")
        cv2.imshow(win, disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("m"): 
            warp_state["warp"] = (warp_state["warp"] + 1) % len(WARPS)
            dirty = True
        elif key == ord("d"): 
            warp_state["metric"] = (warp_state["metric"] + 1) % len(METRICS)
            dirty = True
        elif key == ord("r"): 
            warp_state["ripple"] = not warp_state["ripple"]
            dirty = True
        elif key == ord("s"): 
            cv2.imwrite(f"../data/{name}_{METRICS[warp_state['metric']]}_{WARPS[warp_state['warp']]}.png", warped)
        elif key == ord("b"):
            bake()

        elif key in (27, ord("q")): break

    cv2.destroyAllWindows()


#MAIN FUNCTION
def main():
    parser = ap.ArgumentParser(description="MLM Python Demo")
    parser.add_argument("path", type=str, help="Path to image file.")
    parser.add_argument("-o", "--output_name", default=None, type=str, help="name of output file (without filetype)")
    parser.add_argument("-k", "--kap_hyp", default=2.5, type=np.float32, help="Number of hyperbolic unit distances the image's screen half-diagonal should cover")
    parser.add_argument("-r", "--grid_res", default=10, type=int, help="Number of pixels per grid square axis")
    parser.add_argument("-c", "--canvas", default=800, type=int, help="Number of pixels per canvas axis")

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
    name = args.output_name if args.output_name is not None else img_path.stem
    points = []
    place_points(img, points)
    #compute constants from control points
    if(len(points) == 0):
        print("adding center point")
        points.append((w//2, h//2))
    P = np.array(points).astype(np.float32)
    move_points(img, P, name,K=args.kap_hyp,S=args.canvas,grid_res=args.grid_res)

if __name__ == "__main__":
    main()