import dearpygui.dearpygui as dpg, cv2, numpy as np
from pathlib import Path
from mls import WarpEngine, WARPS, METRICS

S, SIDEBAR_W = 800, 260
editor, out_dir, stem = None, Path.home(), "output"

def to_rgba(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGBA).astype(np.float32).ravel() / 255.0

# ---------- callbacks ----------
def on_open(sender, app_data):
    global editor, stem
    path = app_data["file_path_name"]; img = cv2.imread(path)
    if img is None: return
    stem = Path(path).stem
    editor = WarpEngine(img, S=S) if editor is None else (editor.load_img(img) or editor)

def on_outdir(sender, app_data):
    global out_dir; out_dir = Path(app_data["file_path_name"]); dpg.set_value("outdir_txt", str(out_dir))

def on_save():
    if editor: cv2.imwrite(str(out_dir / f"{stem}.png"), editor.render_full())

def canvas_xy():  # mouse in canvas pixels, accounting for display scaling
    mx, my = dpg.get_mouse_pos(local=False)
    x0, y0 = dpg.get_item_rect_min("canvas_img"); dw, _ = dpg.get_item_rect_size("canvas_img")
    k = S / max(dw, 1); return (mx - x0) * k, (my - y0) * k

def on_click(): 
    if editor and dpg.is_item_hovered("canvas_img"): editor.mouse_down(*canvas_xy())

def on_move():  
    if editor: editor.mouse_move(*canvas_xy())

def on_release():
    if editor: editor.mouse_up()

def on_reset():
    if editor: editor.reset_points()

def layout():  # center the canvas in whatever space is left beside the sidebar
    vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
    side = max(100, min(vw - SIDEBAR_W - 30, vh - 30))
    dpg.configure_item("canvas_img", width=side, height=side)
    dpg.set_item_pos("canvas_img", [SIDEBAR_W + (vw - SIDEBAR_W - side) // 2, (vh - side) // 2])

#SETUP
dpg.create_context()

with dpg.texture_registry():
    dpg.add_raw_texture(S, S, np.zeros(S * S * 4, np.float32), format=dpg.mvFormat_Float_rgba, tag="canvas_tex")

with dpg.file_dialog(show=False, callback=on_open, tag="open_dlg", width=600, height=400):
    dpg.add_file_extension("Images{.png,.jpg,.jpeg,.bmp,.webp}")
dpg.add_file_dialog(directory_selector=True, show=False, callback=on_outdir, tag="dir_dlg", width=600, height=400)

with dpg.window(tag="main"):
    with dpg.child_window(width=SIDEBAR_W, pos=[0, 0], resizable_x=True, tag="sidebar"):
        dpg.add_button(label="Open image", callback=lambda: dpg.show_item("open_dlg"))
        dpg.add_button(label="Output folder", callback=lambda: dpg.show_item("dir_dlg"))
        dpg.add_text(str(out_dir), tag="outdir_txt", wrap=SIDEBAR_W - 20)
        dpg.add_separator()
        dpg.add_combo(WARPS, default_value=WARPS[0], label="Warp",
                      callback=lambda s, a: editor and editor.set_warp(WARPS.index(a)))
        dpg.add_combo(METRICS, default_value=METRICS[0], label="Distance",
                      callback=lambda s, a: editor and editor.set_metric(METRICS.index(a)))
        dpg.add_checkbox(label="Ripple", callback=lambda s, a: editor and editor.set_ripple(a))
        dpg.add_checkbox(label="Place points", default_value=True, callback=lambda s, a: editor and editor.set_placing(a))
        dpg.add_slider_float(label="Lambda", min_value=0, max_value=1,
                             callback=lambda s, a: editor and editor.set_lambda(a))
        dpg.add_separator()
        dpg.add_button(label="Bake", callback=lambda: editor and editor.bake())
        dpg.add_button(label="Save", callback=on_save)
        dpg.add_button(label="Reset", callback=on_reset)
        dpg.add_button(label="Toggle fullscreen", callback=lambda: dpg.toggle_viewport_fullscreen())
    dpg.add_image("canvas_tex", tag="canvas_img")

with dpg.handler_registry():
    dpg.add_mouse_click_handler(button=dpg.mvMouseButton_Left, callback=on_click)
    dpg.add_mouse_move_handler(callback=on_move)
    dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=on_release)
    dpg.add_key_press_handler(dpg.mvKey_F, callback=lambda: dpg.toggle_viewport_fullscreen())

dpg.create_viewport(title="MLS PLUS", width=SIDEBAR_W + S + 40, height=S + 40)
dpg.set_viewport_resize_callback(lambda: layout())
dpg.setup_dearpygui(); dpg.show_viewport(); dpg.set_primary_window("main", True); layout()


#RENDER LOOP
while dpg.is_dearpygui_running():
    if editor is not None and (frame := editor.tick()) is not None:
        dpg.set_value("canvas_tex", to_rgba(frame))
    dpg.render_dearpygui_frame()
dpg.destroy_context()