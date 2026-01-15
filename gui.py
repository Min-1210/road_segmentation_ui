#!/usr/bin/env python3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import os
import sys

# Try importing the inference logic
try:
    from inference import build_model, overlay_mask
except ImportError:
    # Add current directory to sys.path for Linux execution
    sys.path.append(os.getcwd())
    try:
        from inference import build_model, overlay_mask
    except ImportError:
        print("ERROR: Could not find 'inference.py'. Please ensure it is in the same directory.")

class RoadSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Road Segmentation Tool")
        self.root.geometry("1250x800")

        # Variables
        self.original_cv_image = None
        self.processed_cv_image = None
        self.result_pil_image = None
        self.model = None
        self.weight_path = ""
        
        # Check Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on device: {self.device}")
        
        # Layout Frames
        self.left_frame = tk.Frame(root, width=350, bg="#f0f0f0", padx=10, pady=10)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.right_frame = tk.Frame(root, bg="white")
        self.right_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # Cross-platform fonts
        self.header_font = ("Sans", 10, "bold")
        self.normal_font = ("Sans", 10)

        self._init_controls()
        self._init_display()

    def _init_controls(self):
        # --- Section 1: Input ---
        tk.Label(self.left_frame, text="1. INPUT IMAGE", font=self.header_font, bg="#f0f0f0").pack(anchor="w")
        tk.Button(self.left_frame, text="📂 Open Image", command=self.load_image, bg="#ADD8E6", width=25).pack(pady=5)
        
        tk.Label(self.left_frame, text="-"*40, bg="#f0f0f0").pack()
        
        # --- Section 2: Image Processing ---
        tk.Label(self.left_frame, text="2. IMAGE PROCESSING (OpenCV)", font=self.header_font, bg="#f0f0f0").pack(anchor="w", pady=(10, 5))
        
        # Blur
        tk.Label(self.left_frame, text="Blur:", bg="#f0f0f0", font=self.normal_font).pack(anchor="w")
        frame_blur = tk.Frame(self.left_frame, bg="#f0f0f0")
        frame_blur.pack(fill=tk.X)
        self.blur_val = tk.IntVar(value=0)
        scale_blur = tk.Scale(frame_blur, from_=0, to=10, orient=tk.HORIZONTAL, variable=self.blur_val, bg="#f0f0f0", highlightthickness=0)
        scale_blur.pack(side=tk.LEFT, expand=True, fill=tk.X)
        entry_blur = tk.Entry(frame_blur, textvariable=self.blur_val, width=5, justify='center')
        entry_blur.pack(side=tk.RIGHT, padx=5)

        # Brightness/Alpha
        tk.Label(self.left_frame, text="Brightness (Alpha %):", bg="#f0f0f0", font=self.normal_font).pack(anchor="w", pady=(5, 0))
        frame_alpha = tk.Frame(self.left_frame, bg="#f0f0f0")
        frame_alpha.pack(fill=tk.X)
        self.alpha_val = tk.IntVar(value=100)
        scale_alpha = tk.Scale(frame_alpha, from_=0, to=200, orient=tk.HORIZONTAL, variable=self.alpha_val, bg="#f0f0f0", highlightthickness=0)
        scale_alpha.pack(side=tk.LEFT, expand=True, fill=tk.X)
        entry_alpha = tk.Entry(frame_alpha, textvariable=self.alpha_val, width=5, justify='center')
        entry_alpha.pack(side=tk.RIGHT, padx=5)

        # Gaussian Noise
        self.noise_var = tk.BooleanVar()
        tk.Checkbutton(self.left_frame, text="Add Gaussian Noise", variable=self.noise_var, bg="#f0f0f0", highlightthickness=0).pack(anchor="w", pady=(5, 0))
        
        # Apply Button
        tk.Button(self.left_frame, text="⚡ Apply Effects", command=self.apply_noise_effects).pack(pady=10, fill=tk.X)
        
        tk.Label(self.left_frame, text="-"*40, bg="#f0f0f0").pack()

        # --- Section 3: Model Config ---
        tk.Label(self.left_frame, text="3. MODEL CONFIGURATION", font=self.header_font, bg="#f0f0f0").pack(anchor="w", pady=(10, 5))

        tk.Label(self.left_frame, text="Architecture:", bg="#f0f0f0").pack(anchor="w")
        self.arch_options = ["DeepLabV3Plus", "FPN", "MANet", "PAN", "PSPNet", "UPerNet", "EfficientViT-Seg"]
        self.combo_arch = ttk.Combobox(self.left_frame, values=self.arch_options)
        self.combo_arch.current(0)
        self.combo_arch.pack(fill=tk.X, pady=2)

        tk.Label(self.left_frame, text="Encoder:", bg="#f0f0f0").pack(anchor="w")
        self.encoder_options = ["mobileone_s0", "mobileone_s1", "mobileone_s2", "mobileone_s3", "efficientvit-seg-l1-ade20k"]
        self.combo_encoder = ttk.Combobox(self.left_frame, values=self.encoder_options)
        self.combo_encoder.current(0)
        self.combo_encoder.pack(fill=tk.X, pady=2)

        tk.Label(self.left_frame, text="Model Weights (.pt):", bg="#f0f0f0").pack(anchor="w")
        self.btn_weight = tk.Button(self.left_frame, text="... Select File", command=self.select_weight)
        self.btn_weight.pack(fill=tk.X, pady=2)
        
        self.lbl_weight_name = tk.Label(self.left_frame, text="No file selected", fg="gray", bg="#f0f0f0", wraplength=300, font=("Sans", 8, "italic"))
        self.lbl_weight_name.pack(anchor="w")

        tk.Label(self.left_frame, text="-"*40, bg="#f0f0f0").pack()

        # --- Section 4: Actions ---
        tk.Button(self.left_frame, text="🚀 RUN INFERENCE", command=self.run_inference, bg="green", fg="white", font=("Sans", 11, "bold"), height=2).pack(fill=tk.X, pady=10)
        tk.Button(self.left_frame, text="💾 Save Result", command=self.save_result, bg="orange").pack(fill=tk.X)
        tk.Button(self.left_frame, text="🔄 Reset", command=self.reset_app, bg="gray", fg="white").pack(fill=tk.X, pady=20)

    def _init_display(self):
        # Input Preview
        self.frame_in = tk.LabelFrame(self.right_frame, text="Input Preview", bg="white")
        self.frame_in.pack(side=tk.TOP, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.lbl_in = tk.Label(self.frame_in, text="Empty", bg="#ddd")
        self.lbl_in.pack(expand=True, fill=tk.BOTH)
        
        # Output Preview
        self.frame_out = tk.LabelFrame(self.right_frame, text="Inference Result", bg="white")
        self.frame_out.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.lbl_out = tk.Label(self.frame_out, text="Empty", bg="#ddd")
        self.lbl_out.pack(expand=True, fill=tk.BOTH)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.bmp")])
        if not path: return
        try:
            img = cv2.imread(path)
            if img is None: raise Exception("Could not read image file.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.original_cv_image = img
            self.processed_cv_image = img.copy()
            self.show_image(img, self.lbl_in)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_image(self, cv_img, label):
        h, w = cv_img.shape[:2]
        display_h = 300
        scale = display_h/h
        new_w, new_h = int(w*scale), display_h
        
        # Use LANCZOS for best quality
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS 

        resized = cv2.resize(cv_img, (new_w, new_h))
        tk_img = ImageTk.PhotoImage(Image.fromarray(resized))
        label.config(image=tk_img, text="")
        label.image = tk_img

    def apply_noise_effects(self):
        if self.original_cv_image is None: return
        
        try:
            k_val = self.blur_val.get()
            alpha_val = self.alpha_val.get()
        except tk.TclError:
            messagebox.showwarning("Input Error", "Please enter valid integers.")
            return

        img = self.original_cv_image.copy()

        # 1. Blur
        if k_val > 0:
            k = k_val * 2 + 1
            img = cv2.GaussianBlur(img, (k, k), 0)

        # 2. Brightness/Alpha
        alpha_scale = alpha_val / 100.0
        if alpha_scale != 1.0:
            img = cv2.convertScaleAbs(img, alpha=alpha_scale, beta=0)

        # 3. Noise
        if self.noise_var.get():
            gauss = np.random.normal(0, 0.5**0.5, img.shape) * 50
            img = np.clip(img.astype(np.float32) + gauss, 0, 255).astype(np.uint8)

        self.processed_cv_image = img
        self.show_image(img, self.lbl_in)

    def select_weight(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pt *.pth")])
        if path:
            self.weight_path = path
            self.lbl_weight_name.config(text=os.path.basename(path), fg="black")

    def run_inference(self):
        if self.processed_cv_image is None or not self.weight_path:
            messagebox.showwarning("Warning", "Please select an image and a model weight file.")
            return
        
        # Determine cursor type based on OS
        loading_cursor = "wait" if os.name == 'nt' else "watch"
        self.root.config(cursor=loading_cursor)
        self.root.update()
        
        try:
            arch = self.combo_arch.get()
            enc = self.combo_encoder.get()
            
            print(f"-> Loading model: {arch} | {enc}")
            self.model = build_model(arch, enc, num_classes=2, weight_path=self.weight_path, device=self.device)
            self.model.eval()

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            t_img = transform(Image.fromarray(self.processed_cv_image)).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(t_img)
                if out.shape[-2:] != t_img.shape[-2:]:
                    out = F.interpolate(out, size=t_img.shape[-2:], mode='bilinear', align_corners=False)
                mask = torch.argmax(out, dim=1).long().squeeze().cpu().numpy()

            res = overlay_mask(self.processed_cv_image, mask)
            self.result_pil_image = Image.fromarray(res)
            self.show_image(res, self.lbl_out)
            
            print("-> Inference complete.")
            
        except Exception as e:
            print(f"Error: {e}")
            messagebox.showerror("Inference Error", str(e))
        finally:
            self.root.config(cursor="")

    def save_result(self):
        if self.result_pil_image:
            p = filedialog.asksaveasfilename(defaultextension=".png", 
                                           filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
            if p: 
                self.result_pil_image.save(p)
                messagebox.showinfo("Saved", f"Image saved successfully to:\n{p}")

    def reset_app(self):
        self.original_cv_image = None
        self.processed_cv_image = None
        self.result_pil_image = None
        
        self.lbl_in.config(image='', text="Empty")
        self.lbl_out.config(image='', text="Empty")
        
        self.blur_val.set(0)
        self.alpha_val.set(100)
        self.noise_var.set(False)

if __name__ == "__main__":
    # Environment check for Linux (SSH/Headless)
    if os.name == 'posix' and os.environ.get('DISPLAY', '') == '':
        print('WARNING: No DISPLAY variable found. Are you running in a headless environment?')
    
    root = tk.Tk()
    try:
        app = RoadSegmentationApp(root)
        root.mainloop()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")