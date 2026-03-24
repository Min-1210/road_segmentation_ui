import os
import threading
import sys
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps

import torch
import torch.nn.functional as F
from torchvision import transforms

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from inference import build_model, overlay_mask
except ImportError:
    def build_model(*args): return torch.nn.Identity()
    def overlay_mask(img, mask, alpha): return img

try:
    import gui_config
except ImportError:
    gui_config = None
    print("Warning: gui_config.py not found. Auto-load features disabled.")

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def is_image_file(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in VALID_EXTS

def cv2_imread_rgb(path: str) -> np.ndarray:
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

class InteractiveCanvas(tk.Canvas):
    def __init__(self, parent, main_app=None, **kwargs):
        super().__init__(parent, bg="#f0f0f0", highlightthickness=0, **kwargs)
        self.main_app = main_app
        self.orig_img = None
        self.tk_img = None
        self.zoom = -1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_data = {"x": 0, "y": 0}

        self.bind("<MouseWheel>", self.do_zoom)
        self.bind("<Button-4>", self.do_zoom)
        self.bind("<Button-5>", self.do_zoom)
        self.bind("<ButtonPress-1>", self.start_pan)
        self.bind("<B1-Motion>", self.do_pan)
        self.bind("<Configure>", self.on_resize)
        
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        if self.main_app and hasattr(self.main_app, 'canvas'):
            self.main_app.canvas.unbind_all("<MouseWheel>")
            self.main_app.canvas.unbind_all("<Button-4>")
            self.main_app.canvas.unbind_all("<Button-5>")

    def on_leave(self, event):
        if self.main_app and hasattr(self.main_app, '_bind_mouse_scroll'):
            self.main_app._bind_mouse_scroll(self.main_app.canvas)

    def set_image(self, numpy_img):
        if numpy_img is None:
            self.orig_img = None
            self.redraw()
            return
            
        self.orig_img = Image.fromarray(numpy_img)
        self.zoom = -1.0
        self.pan_x = 0
        self.pan_y = 0
        self.redraw()

    def on_resize(self, event):
        self.redraw()

    def _calc_fit_zoom(self):
        if self.orig_img is None: return
        w, h = self.orig_img.size
        
        self.update_idletasks() 
        cw, ch = self.winfo_width(), self.winfo_height()
        
        if cw > 10 and ch > 10:
            self.zoom = min(cw/w, ch/h) * 0.95
        else:
            self.zoom = 1.0 

    def do_zoom(self, event):
        if self.orig_img is None: return
        if self.zoom == -1.0:
            self._calc_fit_zoom()

        if event.num == 4 or event.delta > 0:
            self.zoom *= 1.15
        elif event.num == 5 or event.delta < 0:
            self.zoom /= 1.15
            
        self.zoom = max(0.05, min(self.zoom, 20.0))
        self.redraw()

    def start_pan(self, event):
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def do_pan(self, event):
        if self.orig_img is None: return
        dx = event.x - self.drag_data["x"]
        dy = event.y - self.drag_data["y"]
        self.pan_x += dx
        self.pan_y += dy
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y
        self.redraw()

    def redraw(self):
        self.delete("all")
        if self.orig_img is None:
            self.create_text(self.winfo_width()//2, self.winfo_height()//2, text="No Image", fill="black", font=("Arial", 14))
            return
            
        if self.zoom == -1.0:
            self._calc_fit_zoom()

        w, h = self.orig_img.size
        valid_zoom = max(0.01, self.zoom if self.zoom > 0 else 1.0)
        
        new_w, new_h = max(1, int(w * valid_zoom)), max(1, int(h * valid_zoom))
        
        try: 
            resample = Image.Resampling.NEAREST if valid_zoom > 3 else Image.Resampling.LANCZOS
        except: 
            resample = Image.NEAREST
            
        resized = self.orig_img.resize((new_w, new_h), resample)
        self.tk_img = ImageTk.PhotoImage(resized)
        
        cw, ch = self.winfo_width(), self.winfo_height()
        x = cw//2 + self.pan_x
        y = ch//2 + self.pan_y
        
        self.create_image(x, y, anchor="center", image=self.tk_img)

class CombinedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Road Segmentation GUI Tool")
        self.geometry("1350x900")

        self.input_mode = tk.StringVar(value="single")
        self.input_path = tk.StringVar(value="")
        
        self.arch = tk.StringVar(value="Choose Architecture")
        self.encoder = tk.StringVar(value="Choose Encoder")
        self.dataset = tk.StringVar(value="Choose Dataset")
        self.weight_path = tk.StringVar(value="")
        self.num_classes = tk.IntVar(value=2)
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        
        self.output_dir = tk.StringVar(value=os.path.abspath("predictions"))
        self.overlay_alpha = tk.DoubleVar(value=0.5) 
        
        self.blur_val = tk.IntVar(value=0)       
        self.alpha_val = tk.IntVar(value=100)    
        self.noise_var = tk.BooleanVar(value=False)

        self.save_enabled = tk.BooleanVar(value=True)
        self.current_image_path = None
        self.image_list = []
        self.original_image_data = None
        
        self.last_result_overlay = None 
        self.last_result_mask = None
        
        self.model_cache = {}
        self.model = None
        self.q = queue.Queue() 

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._build_ui()
        self._bind_events()
        self._init_combobox_data()
        self.after(100, self._poll_queue)

        style = ttk.Style()

        self.last_result_mask = None
        self.result_cache = {}

        # style.configure(".", font=("Helvetica", 12)) 
        # style.configure("TButton", padding=10) 
        # style.configure("TEntry", padding=5)

    def _init_combobox_data(self):
        if gui_config and hasattr(gui_config, 'MODEL_DB'):
            self.cb_arch['values'] = list(gui_config.MODEL_DB.keys())
        else:
            self.cb_arch['values'] = ["DeepLabV3Plus", "EfficientViT-Seg"]

    def _build_ui(self):
        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_outer_frame = ttk.Frame(paned, padding=5)
        
        right_frame = ttk.Frame(paned, padding=10)
        
        paned.add(left_outer_frame, weight=0) 
        paned.add(right_frame, weight=1)

        action_area = ttk.Frame(left_outer_frame)
        action_area.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
        self._build_action_bar(action_area)

        self.canvas = tk.Canvas(left_outer_frame, highlightthickness=0, width=550)
        scrollbar = ttk.Scrollbar(left_outer_frame, orient="vertical", command=self.canvas.yview)
        
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        frame_id = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfig(frame_id, width=e.width))

        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._bind_mouse_scroll(self.canvas)

        self._build_input_group(self.scrollable_frame)
        self._build_custom_augment_group(self.scrollable_frame)
        self._build_model_group(self.scrollable_frame)
        self._build_output_group(self.scrollable_frame)
        
        self._build_preview_panel(right_frame)

    def _bind_mouse_scroll(self, widget):
        def _on_mousewheel(event):
            widget.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _on_linux_scroll_up(event):
            widget.yview_scroll(-1, "units")
            
        def _on_linux_scroll_down(event):
            widget.yview_scroll(1, "units")

        widget.bind_all("<MouseWheel>", _on_mousewheel)
        widget.bind_all("<Button-4>", _on_linux_scroll_up)
        widget.bind_all("<Button-5>", _on_linux_scroll_down)

    def _build_input_group(self, parent):
        gb = ttk.LabelFrame(parent, text="1. Select Input Image(s)")
        gb.pack(fill=tk.X, pady=(0, 10))

        row = ttk.Frame(gb)
        row.pack(fill=tk.X, pady=2)
        ttk.Radiobutton(row, text="Single Image", variable=self.input_mode, value="single", command=self._toggle_input_mode).pack(side=tk.LEFT)
        ttk.Radiobutton(row, text="Entire Folder", variable=self.input_mode, value="folder", command=self._toggle_input_mode).pack(side=tk.LEFT, padx=10)

        path_row = ttk.Frame(gb)
        path_row.pack(fill=tk.X, pady=5)
        ttk.Entry(path_row, textvariable=self.input_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(path_row, text="Open...", command=self.on_browse_input).pack(side=tk.LEFT, padx=5)

        frame_list = ttk.Frame(gb)
        frame_list.pack(fill=tk.X, pady=5)
        
        scrollbar = ttk.Scrollbar(frame_list, orient="vertical")
        
        self.lb_images = tk.Listbox(frame_list, height=8, yscrollcommand=scrollbar.set)
        
        scrollbar.config(command=self.lb_images.yview)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.lb_images.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.lb_info = ttk.Label(gb, text="Please select an image.", foreground="blue")
        self.lb_info.pack(fill=tk.X)


        def _on_enter(event):
            self.canvas.unbind_all("<MouseWheel>")
            self.canvas.unbind_all("<Button-4>")
            self.canvas.unbind_all("<Button-5>")

        def _on_leave(event):
            self._bind_mouse_scroll(self.canvas)

        self.lb_images.bind("<Enter>", _on_enter)
        self.lb_images.bind("<Leave>", _on_leave)

    def _build_custom_augment_group(self, parent):
        gb = ttk.LabelFrame(parent, text="2. Image Processing")
        gb.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(gb, text="Blur:").pack(anchor="w")
        row_blur = ttk.Frame(gb)
        row_blur.pack(fill=tk.X)
        
        tk.Scale(row_blur, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.blur_val, bg="#f0f0f0", highlightthickness=0, command=self._debounced_preview).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Entry(row_blur, textvariable=self.blur_val, width=5).pack(side=tk.RIGHT, padx=5)

        ttk.Label(gb, text="Brightness (Alpha %):").pack(anchor="w", pady=(5, 0))
        row_alpha = ttk.Frame(gb)
        row_alpha.pack(fill=tk.X)
        
        tk.Scale(row_alpha, from_=0, to=200, orient=tk.HORIZONTAL, variable=self.alpha_val, bg="#f0f0f0", highlightthickness=0, command=self._debounced_preview).pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Entry(row_alpha, textvariable=self.alpha_val, width=5).pack(side=tk.RIGHT, padx=5)

        ttk.Checkbutton(gb, text="Add Gaussian Noise", variable=self.noise_var, command=self._debounced_preview).pack(anchor="w", pady=(5, 0))
        ttk.Button(gb, text="Reset Image Processing", command=self.on_reset_filters).pack(fill=tk.X, pady=5)

    def _build_model_group(self, parent):
        gb = ttk.LabelFrame(parent, text="3. Model Configuration")
        gb.pack(fill=tk.X, pady=(0, 10))

        grid = ttk.Frame(gb)
        grid.pack(fill=tk.X, pady=5)
        
        ttk.Label(grid, text="Architecture:").grid(row=0, column=0, sticky="w")
        self.cb_arch = ttk.Combobox(grid, textvariable=self.arch, state="readonly", width=22)
        self.cb_arch.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        ttk.Label(grid, text="Encoder:").grid(row=1, column=0, sticky="w")
        self.cb_encoder = ttk.Combobox(grid, textvariable=self.encoder, state="readonly", width=22)
        self.cb_encoder.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(grid, text="Dataset:").grid(row=2, column=0, sticky="w")
        self.cb_dataset = ttk.Combobox(grid, textvariable=self.dataset, state="readonly", width=22)
        self.cb_dataset.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(gb, text="File Weight (.pt):").pack(anchor="w", pady=(5,0))
        row_w = ttk.Frame(gb)
        row_w.pack(fill=tk.X)
        ttk.Entry(row_w, textvariable=self.weight_path, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row_w, text="...", width=4, command=self.on_browse_weight).pack(side=tk.LEFT, padx=2)
        
        self.lb_model_status = ttk.Label(gb, text="Model not loaded", foreground="gray")
        self.lb_model_status.pack(anchor="w")

        ttk.Button(gb, text="Load Model", command=self.on_load_model).pack(fill=tk.X, pady=5)

    def _build_output_group(self, parent):
        gb = ttk.LabelFrame(parent, text="4. Output & Save")
        gb.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(gb, text="Auto Save to folder", variable=self.save_enabled).pack(anchor="w")
        
        row = ttk.Frame(gb)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self.output_dir).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(row, text="...", width=4, command=self.on_browse_output).pack(side=tk.LEFT)

        ttk.Separator(gb, orient='horizontal').pack(fill='x', pady=5)
        self.btn_manual_save = ttk.Button(gb, text="Save Current Result Manually...", command=self.on_manual_save, state="disabled")
        self.btn_manual_save.pack(fill=tk.X, pady=(0, 5))

    def _build_action_bar(self, parent):
        self.btn_run = tk.Button(parent, text="RUN DETECT", command=self.on_run, bg="green", fg="white", font=("Arial", 12, "bold"), height=2)
        self.btn_run.pack(fill=tk.X, pady=5)
        ttk.Button(parent, text="Reset", command=self.on_reset).pack(fill=tk.X)

    def _build_preview_panel(self, parent):    
        split_view = ttk.PanedWindow(parent, orient=tk.HORIZONTAL)
        split_view.pack(fill=tk.BOTH, expand=True)
        
        frame_input = ttk.LabelFrame(split_view, text="Input Image")
        split_view.add(frame_input, weight=1)
        
        self.lbl_input = InteractiveCanvas(frame_input, main_app=self)
        self.lbl_input.pack(fill=tk.BOTH, expand=True)
        
        frame_output = ttk.LabelFrame(split_view, text="Segmentation Result")
        split_view.add(frame_output, weight=1)
        
        self.lbl_output = InteractiveCanvas(frame_output, main_app=self)
        self.lbl_output.pack(fill=tk.BOTH, expand=True)

        self.preview_labels = {
            "Input Preview": self.lbl_input,
            "Result (Overlay)": self.lbl_output,
            "Mask Only": None 
        }

        log_frame = ttk.LabelFrame(parent, text="Log")
        log_frame.pack(fill=tk.X, pady=(5, 0))
        self.txt_log = tk.Text(log_frame, height=5, state="disabled", bg="#f0f0f0")
        self.txt_log.pack(fill=tk.X)

    def log(self, msg):
        self.txt_log.configure(state="normal")
        self.txt_log.insert(tk.END, ">> " + msg + "\n")
        self.txt_log.see(tk.END)
        self.txt_log.configure(state="disabled")

    def _bind_events(self):
        self.cb_arch.bind("<<ComboboxSelected>>", self._on_arch_changed)
        self.cb_encoder.bind("<<ComboboxSelected>>", self._on_encoder_changed)
        self.cb_dataset.bind("<<ComboboxSelected>>", self._auto_select_weight)
        self.lb_images.bind("<<ListboxSelect>>", self.on_select_listbox)
            
    def _debounced_preview(self, *args):
        if not hasattr(self, '_preview_timer'):
            self._preview_timer = None
            
        if self._preview_timer is not None:
            self.after_cancel(self._preview_timer)
            
        self._preview_timer = self.after(150, self.on_preview_noise)

    def _on_arch_changed(self, event):
        arch = self.arch.get()
        if gui_config and arch in gui_config.MODEL_DB:
            encoders = list(gui_config.MODEL_DB[arch].keys())
            self.cb_encoder['values'] = encoders
            if encoders:
                self.cb_encoder.current(0)
                self._on_encoder_changed(None)
        else:
            self.cb_encoder['values'] = []

    def _on_encoder_changed(self, event):
        arch = self.arch.get()
        enc = self.encoder.get()
        
        current_selected_dataset = self.dataset.get()

        if gui_config:
            datasets = gui_config.get_available_datasets(arch, enc)
            self.cb_dataset['values'] = datasets
            
            if datasets:
                if current_selected_dataset in datasets:
                    self.cb_dataset.set(current_selected_dataset)
                else:
                    self.cb_dataset.current(0)
                
                self._auto_select_weight(None)
            else:
                self.cb_dataset.set("No dataset found")
                self.weight_path.set("")

    def _auto_select_weight(self, event=None):
        if not gui_config: return
        path = gui_config.get_weight_path(self.arch.get(), self.encoder.get(), self.dataset.get())
        if path and os.path.exists(path):
            self.weight_path.set(path)
            self.lb_model_status.config(text=f"Auto: {os.path.basename(path)}", foreground="blue")
        else:
            self.lb_model_status.config(text="Weight file not found", foreground="orange")

    def _toggle_input_mode(self):
        self.lb_images.delete(0, tk.END)
        self.input_path.set("")
        self.image_list = []
        self.current_image_path = None
        self.lb_info.config(text="Please select an image.")

    def on_browse_input(self):
        mode = self.input_mode.get()
        if mode == "single":
            path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
            if path:
                self.input_path.set(path)
                self.image_list = [path]
                self.lb_images.delete(0, tk.END)
                self.lb_images.insert(tk.END, os.path.basename(path))
                self.lb_images.selection_set(0)
                self.current_image_path = path
                self.lb_info.config(text=f"Selecting: {os.path.basename(path)}")
                self._load_image_preview()
        else:
            folder = filedialog.askdirectory()
            if folder:
                self.input_path.set(folder)
                files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if is_image_file(f)])
                self.image_list = files
                self.lb_images.delete(0, tk.END)
                for f in files:
                    self.lb_images.insert(tk.END, os.path.basename(f))
                self.lb_info.config(text=f"Found {len(files)} images.")
                if files:
                    self.lb_images.selection_set(0)
                    self.current_image_path = files[0]
                    self._load_image_preview()

    def on_select_listbox(self, event):
        sel = self.lb_images.curselection()
        if sel:
            idx = int(sel[0])
            if idx < len(self.image_list):
                self.current_image_path = self.image_list[idx]
                self._load_image_preview()
                if self.current_image_path in self.result_cache:
                    cached_data = self.result_cache[self.current_image_path]
                    overlay_np = cached_data["overlay"]
                    
                    self._display_image(self.preview_labels["Result (Overlay)"], overlay_np)
                    self.last_result_overlay = Image.fromarray(overlay_np)
                    self.btn_manual_save.config(state="normal")
                
                else:
                    current_dataset = self.dataset.get()
                    input_filename = os.path.basename(self.current_image_path)
                    expected_filename = f"pred_{current_dataset}_{input_filename}"
                    expected_path = os.path.join(self.output_dir.get(), expected_filename)

                    if os.path.exists(expected_path):
                        try:
                            saved_pil = Image.open(expected_path)
                            saved_numpy = np.array(saved_pil) 
                            self._display_image(self.preview_labels["Result (Overlay)"], saved_numpy)
                            self.last_result_overlay = saved_pil
                            self.btn_manual_save.config(state="normal")
                        except Exception as e:
                            self._reset_result_view()
                    else:
                        self._reset_result_view()
                        self.btn_manual_save.config(state="disabled")

    def _reset_result_view(self):
        self.last_result_overlay = None
        if "Result (Overlay)" in self.preview_labels:
            canvas_widget = self.preview_labels["Result (Overlay)"]
            if canvas_widget:
                canvas_widget.set_image(None)

    def on_browse_weight(self):
        p = filedialog.askopenfilename(filetypes=[("Model", "*.pt *.pth")])
        if p: self.weight_path.set(p)

    def on_browse_output(self):
        p = filedialog.askdirectory()
        if p: self.output_dir.set(p)

    def on_load_model(self):
        if not self.weight_path.get():
            messagebox.showwarning("Error", "File weight not selected!")
            return
        
        key = (self.arch.get(), self.encoder.get(), self.dataset.get(), self.weight_path.get())
        if key in self.model_cache:
            self.model = self.model_cache[key]
            self.lb_model_status.config(text="Model Loaded (Cached) ", foreground="green")
            return

        self.log(f"Loading model: {key[0]}...")
        self.config(cursor="watch")
        threading.Thread(target=self._load_model_thread, args=(key,), daemon=True).start()

    def _load_model_thread(self, key):
        try:
            dev = torch.device(self.device.get())
            m = build_model(self.arch.get(), self.encoder.get(), self.num_classes.get(), self.weight_path.get(), dev)
            m.eval()
            self.model = m
            self.model_cache[key] = m
            self.after(0, lambda: self.lb_model_status.config(text="Model Loaded OK ✅", foreground="green"))
            self.after(0, lambda: self.log("Load model success."))
        except Exception as e:
            err_msg = str(e) 
            
            self.after(0, lambda: messagebox.showerror("Load Error", err_msg))
            self.after(0, lambda: self.log(f"Load Error: {err_msg}"))
            self.after(0, lambda: self.lb_model_status.config(text="Error ❌", foreground="red"))
        finally:
            self.after(0, lambda: self.config(cursor=""))

    def _apply_custom_noise(self, rgb_img):
        out = rgb_img.copy()
        
        try:
            k_val = self.blur_val.get()
            if k_val > 0:
                k = k_val * 2 + 1
                out = cv2.GaussianBlur(out, (k, k), 0)
        except: pass

        try:
            alpha = self.alpha_val.get() / 100.0
            if alpha != 1.0:
                out = cv2.convertScaleAbs(out, alpha=alpha, beta=0)
        except: pass

        if self.noise_var.get():
            mean = 0
            std_dev = 25
            gaussian_noise = np.random.normal(mean, std_dev, out.shape).astype(np.int16)
            noisy_image = out.astype(np.int16) + gaussian_noise
            out = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return out

    def _load_image_preview(self):
        if not self.current_image_path: return
        try:
            self.original_image_data = cv2_imread_rgb(self.current_image_path)
            self.on_preview_noise()
        except Exception as e:
            self.log(f"Error reading image: {e}")

    def on_preview_noise(self, *args):
        if self.original_image_data is None: return
        
        try:
            rgb = self.original_image_data.copy()
            processed = self._apply_custom_noise(rgb)
            self._display_image(self.preview_labels["Input Preview"], processed)
        except Exception as e: print(e)

    def _display_image(self, canvas_widget, numpy_img):
        if canvas_widget is None: return
        canvas_widget.set_image(numpy_img)

    def on_run(self):
        if not self.image_list:
            messagebox.showwarning("!", "Please select image(s).")
            return
        if not self.model:
            messagebox.showwarning("!", "Model not loaded.")
            return

        self.btn_run.config(state="disabled")
        
        mode = self.input_mode.get()
        if mode == "single":
            targets = [self.current_image_path]
        else:
            targets = self.image_list

        self.log(f"🚀 Start run detect for {len(targets)} picture...")
        threading.Thread(target=self._worker_batch_inference, args=(targets,), daemon=True).start()

    def _worker_batch_inference(self, target_list):
        for i, img_path in enumerate(target_list):
            if not img_path: continue
            try:
                self.q.put({"type": "info", "msg": f"Loading [{i+1}/{len(target_list)}]: {os.path.basename(img_path)}"})
                
                rgb = cv2_imread_rgb(img_path)
                rgb_input = self._apply_custom_noise(rgb)
                
                h_orig, w_orig = rgb_input.shape[:2]
                ALIGN = 32 
                target_h = ((h_orig - 1) // ALIGN + 1) * ALIGN
                target_w = ((w_orig - 1) // ALIGN + 1) * ALIGN
                pad_h = target_h - h_orig
                pad_w = target_w - w_orig
                
                if pad_h > 0 or pad_w > 0:
                    rgb_padded = cv2.copyMakeBorder(rgb_input, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
                else:
                    rgb_padded = rgb_input

                pil_img = Image.fromarray(rgb_padded)
                tensor = self.transform(pil_img).unsqueeze(0).to(torch.device(self.device.get()))

                with torch.no_grad():
                    out = self.model(tensor)
                    if out.shape[-2:] != tensor.shape[-2:]:
                        out = F.interpolate(out, size=tensor.shape[-2:], mode='bilinear', align_corners=False)
                    mask_padded = torch.argmax(out, dim=1).long().squeeze().cpu().numpy()

                mask = mask_padded[:h_orig, :w_orig]

                overlay = overlay_mask(rgb_input, mask, alpha=self.overlay_alpha.get())
                mask_vis = (mask * 255).astype(np.uint8)
                if self.num_classes.get() > 1:
                     mask_vis = self._color_mask(mask)

                saved_path = None
                if self.save_enabled.get():
                    os.makedirs(self.output_dir.get(), exist_ok=True)
                    fname = f"pred_{self.dataset.get()}_{os.path.basename(img_path)}"
                    save_p = os.path.join(self.output_dir.get(), fname)
                    Image.fromarray(overlay).save(save_p)
                    saved_path = save_p

                self.q.put({
                    "type": "success",
                    "img_path": img_path, 
                    "overlay": overlay,
                    "mask_vis": mask_vis,
                    "saved_path": saved_path,
                    "shape_info": f"{w_orig}x{h_orig}"
                })

            except Exception as e:
                self.q.put({"type": "error", "msg": f"Error in {os.path.basename(img_path)}: {str(e)}"})
        
        self.q.put({"type": "batch_done", "msg": "Complete full image recognition!"})

    def _color_mask(self, mask):
        h, w = mask.shape
        col = np.zeros((h, w, 3), dtype=np.uint8)
        colors = [[0,0,0], [0,255,0], [255,0,0], [0,0,255], [255,255,0]] 
        for i in range(1, len(colors)):
            col[mask == i] = colors[i]
        return col

    def _poll_queue(self):
        try:
            while True:
                data = self.q.get_nowait()
                
                if data["type"] == "info":
                    self.log(data["msg"])
                    
                elif data["type"] == "success":
                    img_path = data["img_path"]
                    
                    self.result_cache[img_path] = {
                        "overlay": data["overlay"],
                        "mask_vis": data["mask_vis"]
                    }
                    
                    if data["saved_path"]:
                        self.log(f"Auto saved: {os.path.basename(data['saved_path'])}")
                        
                    if img_path == self.current_image_path:
                        self.last_result_overlay = Image.fromarray(data["overlay"])
                        self.last_result_mask = Image.fromarray(data["mask_vis"])
                        self._display_image(self.preview_labels["Result (Overlay)"], data["overlay"])
                        self.btn_manual_save.config(state="normal")
                        
                elif data["type"] == "error":
                    self.log(f"Error: {data['msg']}")
                    
                elif data["type"] == "batch_done":
                    self.log(data["msg"])
                    self.btn_run.config(state="normal")
                    messagebox.showinfo("Done", "All images have been processed!")
                    
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def on_manual_save(self):
        if self.last_result_overlay is None:
            messagebox.showwarning("Warning", "No result to save!")
            return

        initial_name = f"manual_pred_{os.path.basename(self.current_image_path)}" if self.current_image_path else "result.png"
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            initialfile=initial_name,
            initialdir=self.output_dir.get(),
            title="Save Result As"
        )
        
        if file_path:
            try:
                self.last_result_overlay.save(file_path)
                self.log(f"Manually saved to: {file_path}")
                messagebox.showinfo("Success", "Image saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Could not save file: {e}")

    def on_reset_filters(self):
        self.blur_val.set(0)
        self.alpha_val.set(100)
        self.noise_var.set(False)
        self._debounced_preview()

    def on_reset(self):
        self.lb_images.delete(0, tk.END)
        self.image_list = []
        self.current_image_path = None
        self.original_image_data = None
        self.last_result_overlay = None
        self.last_result_mask = None
        self.last_result_mask = None
        
        self.input_mode.set("single")
        self.input_path.set("")
        self.lb_info.config(text="Please select an image.")
        
        self.arch.set("Choose Architecture")
        self.encoder.set("Choose Encoder")
        self.dataset.set("Choose Dataset")
        self.weight_path.set("")
        
        self.model = None 
        self.lb_model_status.config(text="Model reset", foreground="gray")
        
        self.blur_val.set(0)
        self.alpha_val.set(100)
        self.noise_var.set(False)

        self.result_cache.clear()
        
        self.btn_manual_save.config(state="disabled")
        
        for lbl in self.preview_labels.values():
            if lbl:
                if hasattr(lbl, 'set_image'):
                    lbl.set_image(None)
                else:
                    lbl.configure(image="", text="Reset")
            
        if gui_config:
            self.cb_arch.set("Choose Architecture")
            self.cb_encoder['values'] = []
            self.cb_encoder.set("")
            self.cb_dataset['values'] = []
            self.cb_dataset.set("")

        self.log("RESET COMPLETED")

if __name__ == "__main__":
    app = CombinedApp()
    app.mainloop()