import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import argparse as ap
import cv2
from pathlib import Path
import sys
import math
from typing import Any #to silence linter errors
from grad_aware import d2_geodesic, gradient_graph, compute_cost
import copy

from utils import deform_euclidean, compute_constants_from_points, compute_grid, precompute_weights, inv_bilinear_fast
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
        for i, pt in enumerate(param["to_screen"](param["q"])):
            if math.hypot(x - pt[0], y - pt[1]) <= param["drag radius"]:
                param["selected"] = i; break
        if param["selected"] is None and param["placing"] is True:
            from_screen = param["from_screen"](np.array([x, y], float))
            param["p"].append(from_screen)
            param["q"].append(from_screen)
    elif event == cv2.EVENT_MOUSEMOVE and param["selected"] is not None:
        param["q"][param["selected"]] = param["from_screen"](np.array([x, y], float))
        param["changed"] = True
    elif event == cv2.EVENT_LBUTTONUP:
        param["selected"] = None
        param["changed"] = True

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
    flat_ids = raster_ids.ravel()
    
    #multiply by 2^shift
    fixed = np.round(gt * (1 << shift)).astype(np.int32)
    quads = np.stack([fixed[:-1,:-1],fixed[:-1,1:],fixed[1:,1:],fixed[1:,:-1]],2).reshape(-1,4,2)
    fill_poly = cv2.fillPoly
    for k, quad in enumerate(quads):
        fill_poly(raster_ids, [quad], k, lineType=cv2.LINE_8,shift=shift)

    linear_idx = np.flatnonzero(flat_ids>=0)
    #get coordinate pairs, rows and columns of all covered pixels
    k=flat_ids[linear_idx]
    r,c = np.divmod(k, (gw-1)) #divmod divides for first axis and mods for second 
    ys, xs=np.divmod(linear_idx, w)
    x = np.stack([xs, ys], -1).astype(np.float32)

    gt32 = gt.astype(np.float32)
    A = gt32[r,c]
    B = gt32[r,c+1]
    C = gt32[r+1,c+1]
    D = gt32[r+1,c]
    s, t = inv_bilinear_fast(x,A,B,C,D)
    #since source grid is uniform and axis-aligned, bilinear interpolation is just an affine transformation
    gx = g[0,:,0].astype(np.float32) #x coordinates of original grid at first row
    gy = g[:,0,1].astype(np.float32) #y coordinates of original grid at first column
    map_x = np.full(h*w,-1,np.float32) #initialize x and y coordinate maps
    map_y = np.full(h*w,-1,np.float32)
    map_x[linear_idx] = gx[c] + s*(gx[c+1]-gx[c]) #find corresponding x and y transformed vertices from mapped raster ids and s,t coordinates
    map_y[linear_idx] = gy[r] + t*(gy[r+1]-gy[r])
    return cv2.remap(img, map_x.reshape(h,w), map_y.reshape(h,w), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

def move_points(img,p,name,K=2.5,S=800,min_grid_res=10):
    #store mutable lists for p and q
    #q starts as copy of p
    q = p.copy() 
    plist = p.tolist()
    qlist = q.tolist()

    h, w = img.shape[:2]
    scale = max(h,w) / PX
    ui = S/PX
    win = "Image Deformation with MLS"
    cv2.namedWindow(win)
    grid_res = max(min_grid_res, max(h, w) // 150)
    grid = compute_grid(h, w, grid_res)
    MARGIN = 40
    img_c = np.array([(w - 1) / 2, (h - 1) / 2])
    half_diag = np.hypot(*img_c)
    to_disk = lambda x: image_to_disk(x, img_c, K, half_diag) #convert image coords x to disk coords z
    from_disk = lambda z: disk_to_image(z, img_c, K, half_diag) #convert poincare disk coords z to image coords x
    z_grid = to_disk(grid) 
    z_p = to_disk(p)

    def resample_to_view(scale, S):
        new_img=img if scale == 1 else cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        new_grid=grid * scale
        offset=(S - scale * np.array([w, h])) / 2
        c=np.array([(S - 1) / 2] * 2)
        r_px=S / 2 - 10
        return {'img': new_img, 'grid': new_grid, 'offset': offset, 'c': c, 'r_px': r_px, 'scale': scale, 'S': S }

    warp_view = lambda: resample_to_view(min(1.0, (S - 2 * MARGIN) / max(h, w)), S)
    full_view = lambda: resample_to_view(1.0, max(h, w) + 2 * MARGIN)
    live = warp_view()
    #cache cost map and graph (depends only on img)
    grad_cost = compute_cost(img)   
    graph = gradient_graph(grad_cost)
    graph_stale = True
    geo_d2 = None
    warp_state = dict(metric=0, warp=0, ripple=False, lam=MIN_LAM * half_diag)
    state = {"q": qlist, "p": plist, "changed": True, "selected": None, "drag radius": ui*DRAG_RAD,
             "to_screen": None, "from_screen": None, "placing": False}
    cv2.setMouseCallback(win, click_event_warp, param=state)
    cv2.createTrackbar("Lambda", win, 0, 100, lambda _: None)
    hyp = lambda: METRICS[warp_state["metric"]] == "hyperbolic"
    grad = lambda: METRICS[warp_state["metric"]] == "gradient"
    num_ctrl_points = len(plist)

    def to_screen(x, v):
        x = np.asarray(x, float) #mouse callback passes the qlist (a python list)
        if hyp(): return disk_to_screen(to_disk(x), v["c"], v["r_px"])
        return x * v["scale"] + v["offset"]

    def from_screen(y, v):
        y = np.asarray(y, float)
        if hyp(): return from_disk(screen_to_disk(y, v["c"], v["r_px"]))
        return (y - v["offset"]) / v["scale"]

    def rebuild_p():
        z_p = to_disk(p)
        return z_p

    def rebuild():
        metric = METRICS[warp_state["metric"]]
        warp = WARPS[warp_state["warp"]]
        state["to_screen"] = lambda x: to_screen(x, live)
        state["from_screen"] = lambda y: from_screen(y, live)
        if hyp():
            wt, wn = precompute_weights(z_p, z_grid, "disk", eps=1e-12, ripple=bool(warp_state["ripple"]), K=RIPPLE_K_DISK)
            A, vbl = compute_constants_from_hyperbolic_points(z_p, z_grid, wt, wn, mode=warp)
        else:
            dist_kw = {"polar": dict(center=img_c, lam=warp_state["lam"]), "gradient": dict(dist=geo_d2)}.get(metric, {})
            wt, wn = precompute_weights(p, grid, metric, ripple=bool(warp_state["ripple"]), dist_kw=dist_kw)
            A, vbl = compute_constants_from_points(p, grid, wt, wn, mode=warp)
        return A, wn, vbl
    
    def render(v, A, wn, vbl):
        warp = WARPS[warp_state["warp"]]

        if hyp():
            dest = disk_to_screen(deform_hyperbolic(A, wn, to_disk(q), warp, vbl), v["c"], v["r_px"])
        else:
            dest = deform_euclidean(A, wn, q, warp, vbl) *v["scale"] + v["offset"]
        
        return render_warp(v["img"], v["grid"], dest, (v["S"], v["S"]))

    def load_image(new_img, new_p):
        nonlocal img, p, q, h, w, grid, img_c, half_diag, z_grid, z_p, geo_d2, model, plist, qlist, num_ctrl_points, live, graph, grad_cost, graph_stale
        img = new_img
        h, w = img.shape[:2]
        grid_res = max(min_grid_res, max(h, w) // 150)
        grid = compute_grid(h, w, grid_res)
        live=warp_view()
        p = np.asarray(new_p, float)
        q[:] = p 
        plist = p.tolist()
        qlist = q.tolist()
        grad_cost = compute_cost(img)
        graph = gradient_graph(img)
        if grad():
            geo_d2 = d2_geodesic(p,grid,grad_cost,*graph)
            graph_stale = False
        else:
            graph_stale = True

        state["p"], state["q"] = plist, qlist
        num_ctrl_points = len(plist)
        #recompute image center and half-diagonal based on new dimensions so hyperbolic projection isn't based on old values
        img_c = np.array([(w - 1) / 2, (h - 1) / 2])
        half_diag = np.hypot(*img_c)
        z_grid, z_p = to_disk(grid), to_disk(p)
        model = rebuild()
        state["changed"] = True

    def bake():
        v = full_view()
        out = render(v, *model)
        shown = to_screen(q, v) # handle positions on the canvas
        x, y, bw, bh = cv2.boundingRect(out.any(-1).astype(np.uint8))# crop to bounding box around the content
        load_image(out[y:y + bh, x:x + bw].copy(), shown - [x, y])

    def draw_to_screen():
        disp = render(live, *model)
        for i, pt in enumerate(state["to_screen"](q)):
            color = (0, 0, 255) if i == state["selected"] else (0, 255, 0)
            cv2.circle(disp, tuple(int(c) for c in pt), int(ui*RAD), color, -1)

        if hyp():
            cv2.circle(disp, tuple(int(c) for c in live["c"]), int(live["r_px"]), (90, 90, 90), 1) #render outer rim

        cv2.setWindowTitle(win, f"{WARPS[warp_state['warp']]} | {METRICS[warp_state['metric']]} | ripple "
                            f"{'on' if warp_state['ripple'] else 'off'} | can place? " f"{'yes' if state["placing"] else 'no'}  [m]ode [d]istance [r]ipple [s]ave [q]uit")
        cv2.imshow(win, disp)

    model, dirty= rebuild(), False
    draw_to_screen()
    
    # MAIN RENDER LOOP
    while True:
        #check if gradient mode has been switched on and graph is still stale
        if grad() and graph_stale:
            geo_d2 = d2_geodesic(p, grid, grad_cost, *graph)
            graph_stale = False

        #get trackbar values
        t = cv2.getTrackbarPos("Lambda", win) / 100
        lam = ((1 - t) * MIN_LAM + t * MAX_LAM) * half_diag

        #update state based on trackbar values
        if lam != warp_state["lam"]:
            warp_state["lam"] = lam
            dirty |= METRICS[warp_state["metric"]] == "polar"

        #changes have been made, trigger re-render
        if dirty or state["changed"]:
            if dirty:
                model, dirty = rebuild(), False
            if state["changed"]:
                #update q array
                q = np.array(qlist)
                #set changed to false
                state["changed"] = False
            #only draw when changes have been made
            draw_to_screen()

        key = cv2.waitKey(1) & 0xFF
        #check if new points have been added or removed
        if(num_ctrl_points != len(plist)):
            #if P changed, we need to recompute constants/weights (mark as dirty)
            #AND update P, Q, z_p (poincare disk coords of p) and geo_d2 (cost map for gradient aware distance mode)
            dirty = True
            num_ctrl_points = len(plist)
            p = np.array(plist)
            q = np.array(qlist)
            z_p = rebuild_p()
            if grad():
                geo_d2 = d2_geodesic(p, grid, grad_cost, *graph)
                graph_stale = False
            else:
                graph_stale = True
            print(f"control point added at {p[-1]}")
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
            cv2.imwrite(f"../data/{name}_{METRICS[warp_state['metric']]}_{WARPS[warp_state['warp']]}.png", render(full_view(),*model))
        elif key == ord("b"):
            bake()
        elif key == ord("p"):
            state["placing"] = not state["placing"]
            state["changed"] = True

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
    move_points(img, P, name,K=args.kap_hyp,S=args.canvas,min_grid_res=args.grid_res)

if __name__ == "__main__":
    main()