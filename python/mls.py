import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import cv2
from pathlib import Path
from typing import Any #to silence linter errors
from grad_aware import d2_geodesic, gradient_graph, compute_cost

from utils import deform_euclidean, compute_constants_from_points, compute_grid, precompute_weights, inv_bilinear_fast
from hyperbolic import deform_hyperbolic, screen_to_disk, disk_to_screen, image_to_disk, disk_to_image, compute_constants_from_hyperbolic_points
WARPS   = ["affine", "similarity", "rigid"]
METRICS = ["euclidean", "polar", "gradient", "hyperbolic"]
INTERP = {"nn": cv2.INTER_NEAREST, "bl": cv2.INTER_LINEAR}
MIN_POINTS = {"affine": 3, "similarity": 2, "rigid": 1}
RAD = 2
PX = 250

RIPPLE_K_DISK = 0.3

MIN_HYP_R = 1.10
MAX_HYP_R = 2

MIN_LAM = 0.05
MAX_LAM = 5

class WarpEngine:
    def __init__(self, img, K=2.5, S=800, Smin=250, min_grid_px=10, ripple_k_disk=0.3, margin=40, interp_mode="bl", max_radius = 0.995):
        self.K = K
        self.ui_scale = S / Smin
        self.drag_radius = self.ui_scale * RAD * 2
        self.canvas_size=S
        self.min_grid_px = min_grid_px
        self.warp_i = 0
        self.met_i = 0
        self.ripple = False
        self.placing = True
        self.lam_t = 0.0
        self.selected_i = None
        self.margin = margin
        self.ripple_k_disk = ripple_k_disk
        self.interp = INTERP[interp_mode]
        self.max_radius = max_radius
        self.unselected_color = (0, 255, 0)
        self.selected_color = (0, 0, 255)
        self.load_img(img)
    def load_img(self, img, p=None):
        '''Re-initializes anything that depends on the image.'''
        self.img = img
        self.h, self.w = img.shape[:2] 
        self.grid_res = max(self.min_grid_px, max(self.h, self.w) // 150)
        self.grid = compute_grid(self.h, self.w, self.grid_res)
        self.img_center = np.array([(self.w - 1) / 2, (self.h - 1) / 2])
        self.half_diag = np.hypot(*self.img_center)
        self.disk_grid = self._to_disk(self.grid)
        self.preview = self._make_view(min(1.0, (self.canvas_size - 2 * self.margin) / max(self.h, self.w)), self.canvas_size)
        self.gradient_cost = compute_cost(img)
        self.gradient_graph = gradient_graph(self.gradient_cost)
        self.selected_i = None
        self._set_points_array(np.empty((0,2)) if p is None else p)
    def _set_points_array(self, p):
        """replace control point array and deformed array to match"""
        self.p = np.array(p).astype(np.float32).reshape(-1,2)
        self.q = self.p.copy()
        self._points_changed()
    def _points_changed(self):
        """Called whenever a control point is added or removed. Updates anything derived from 
        the control point set P (Weights, constants A_i, and disk coordinates)"""
        self.disk_p = self._to_disk(self.p)
        self.geo_d2 = None #gradient distance metrics need updating, so mark as stale
        self.dirty = True #dirty means we need re-computation of weights and constants
        self.changed = True #changed means we need to re-render.

    # state property (state? THIS? STATE? Hit the road duuuude, this ain't no state property, this is *PRIVATE* propertuh)
    @property
    def warp(self): return WARPS[self.warp_i]
    @property
    def metric(self): return METRICS[self.met_i]
    @property
    def hyp(self): return self.metric == "hyperbolic"
    @property
    def grad(self): return self.metric == "gradient"
    @property
    def lamb(self): return  ((1 - self.lam_t) * MIN_LAM + self.lam_t * MAX_LAM) * self.half_diag

    def can_warp(self):
        return len(self.p) >= MIN_POINTS[self.warp]
    def sidebar_status(self):
        return dict(warp=self.warp, metric=self.metric, ripple=self.ripple, placing=self.placing, n_points=len(self.p), need_points=max(0, MIN_POINTS[self.warp] - len(self.p)))
    def _to_disk(self, x): return image_to_disk(x, self.img_center, self.K, self.half_diag)
    def _from_disk(self, z): return disk_to_image(z, self.img_center, self.K, self.half_diag)
    def _make_view(self, scale, canvas_size):
        """Resize image to view window's size and recompute dependent values."""
        return {
            "img": self.img if scale == 1 else cv2.resize(self.img, None, fx=scale, fy=scale, interpolation=self.interp),
            "grid": self.grid * scale,
            "offset": (canvas_size - scale * np.array([self.w, self.h])) / 2,
            "disk center": np.array([(canvas_size - 1) / 2] * 2),
            "disk radius":   canvas_size / 2 - 10,
            "scale":  scale, 
            "canvas size": canvas_size
        }
    def _full_view(self):
        """Resize back to full resolution for saving and baking"""
        return self._make_view(1.0, max(self.h, self.w) + 2 * self.margin)
    
    def to_screen(self, x, preview=None):
        """Takes image coordinates x and converts them to screen coordinates s_x, based on the size of the 
        view window. Performs hyperbolic projection if necessary"""
        preview = preview or self.preview
        x = np.asarray(x, float)
        if self.hyp:
            return disk_to_screen(self._to_disk(x), preview["disk center"], preview["disk radius"])
        return x * preview["scale"] + preview["offset"]

    def from_screen(self, x, preview=None):
        preview = preview or self.preview
        x = np.asarray(x, float)
        if self.hyp:
            return self._from_disk(screen_to_disk(x, preview["disk center"], preview["disk radius"], self.max_radius))
        return (x - preview["offset"]) / preview["scale"]
    
    def _rebuild(self):
         """Recomputes weights and constants (A, wn, vbl) from p when necessary"""
         self.dirty = False #only call if dirty is true
         if not self.can_warp():
             self.model = None
             return
         if self.hyp:
             wt, wn = precompute_weights(self.disk_p, self.disk_grid, "disk", eps=1e-12,
                                        ripple=self.ripple, K=self.ripple_k_disk)
             A, vbl = compute_constants_from_hyperbolic_points(self.disk_p, self.disk_grid, wt, wn, mode=self.warp)
         else:
            if self.grad and self.geo_d2 is None:
                self.geo_d2 = d2_geodesic(self.p, self.grid, self.gradient_cost, *self.gradient_graph)
            dist_kw = {"polar":    dict(center=self.img_center, lam=self.lamb),
                       "gradient": dict(dist=self.geo_d2)}.get(self.metric, {})
            wt, wn = precompute_weights(self.p, self.grid, self.metric,
                                        ripple=self.ripple, dist_kw=dist_kw)
            A, vbl = compute_constants_from_points(self.p, self.grid, wt, wn, mode=self.warp)
         self.model = (A, wn, vbl)

    def _sync(self):
        if self.dirty:
            self._rebuild()

    def _render(self, v):
        """Deform the grid with the current q (deformed control points) and rasterize into view v. runs per drag."""
        if self.model is None:
            dest = self.to_screen(self.grid, v)
        else:
            A, wn, vbl = self.model
            if self.hyp:
                z = deform_hyperbolic(A, wn, self._to_disk(self.q), self.warp, vbl)
                dest = disk_to_screen(z, v["disk center"], v["disk radius"])
            else:
                dest = deform_euclidean(A, wn, self.q, self.warp, vbl) * v["scale"] + v["offset"]
        return render_warp(v["img"], v["grid"], dest, (v["canvas size"], v["canvas size"]), self.interp)

    def tick(self):
        """Call once per UI frame. Returns a new BGR canvss_size×canvas_size frame, or None if nothing changed."""
        if not (self.dirty or self.changed):
            return None
        self._sync()
        self.changed = False
        frame = self._render(self.preview)
        for i, pt in enumerate(self.to_screen(self.q)):
            color = self.selected_color if i == self.selected_i else self.unselected_color
            cv2.circle(frame, tuple(int(c) for c in pt), int(self.ui_scale * RAD), color, -1)
        if self.hyp:
            cv2.circle(frame, tuple(int(c) for c in self.preview["disk center"]), int(self.preview["disk radius"]),
                       (90, 90, 90), 1)
        return frame

    def set_warp(self, i): self.warp_i = i;   self.dirty = True
    def set_metric(self, i): self.met_i = i; self.dirty = True
    def set_ripple(self, on): self.ripple = bool(on); self.dirty = True

    def set_lambda(self, t01):
        self.lam_t = float(t01)
        if self.metric == "polar":
            self.dirty = True

    def set_placing(self, on):
        self.placing = bool(on)
        self.changed = True #redraw so ui status changes

    def reset_points(self):
        """Removes all control points."""
        self.selected_i = None
        self._set_points_array(np.empty((0, 2)))

    def mouse_down(self, x, y):
        """Select the nearest control point within the drag_radius. add a point if placing is on and nothing is selected."""
        if len(self.q):
            d = np.hypot(*(self.to_screen(self.q) - [x, y]).T)
            i = int(np.argmin(d))
            if d[i] <= self.drag_radius:
                self.selected_i = i
                self.changed = True #redraw handle color
                return
        if self.placing:
            pt = self.from_screen([x, y])
            self.p = np.vstack([self.p, pt])
            self.q = np.vstack([self.q, pt])
            self._points_changed()

    def mouse_move(self, x, y):
        """Drag the selected handle. Only q changes, so not marked dirty, just redraw."""
        if self.selected_i is not None:
            self.q[self.selected_i] = self.from_screen([x, y])
            self.changed = True

    def mouse_up(self):
        if self.selected_i is not None:
            self.selected_i = None
            self.changed = True #redraw handle color

    def render_full(self):
        """Full-resolution render for saving, with no control points"""
        return self._render(self._full_view())

    def save(self, path):
        self._sync()
        return cv2.imwrite(str(path), self.render_full())

    def default_filename(self, stem):
        return f"{stem}_{self.metric}_{self.warp}.png"

    def bake(self):
        """Make the current deformation the new source image, keeping handles where they are. Grid is flattened."""
        self._sync()
        v = self._full_view()
        out = self._render(v)
        shown = self.to_screen(self.q, v) # handle positions on the full-res canvas
        x, y, bw, bh = cv2.boundingRect(out.any(-1).astype(np.uint8)) # crop to content
        self.selected = None
        self.load_img(out[y:y + bh, x:x + bw].copy(), shown - [x, y])


def render_warp(img, g, gt, out_dims, interp, shift=4):
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
    return cv2.remap(img, map_x.reshape(h,w), map_y.reshape(h,w), interp, borderMode=cv2.BORDER_CONSTANT)

