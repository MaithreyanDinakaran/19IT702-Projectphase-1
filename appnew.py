# app.py
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import time
from threading import Thread
from datetime import datetime

# --- CONFIG (default values; will be overridden by model input if available) ---
MODEL_PATH = "deepfake_mobilenetv2_fast1.h5"
IMG_SIZE = (128, 128)             # will be overwritten to model input size if model loads
SINGLE_DISPLAY_SIZE = (300, 300)
PRIMARY_COLOR = "#00BCD4"
FAKE_COLOR = "#E53935"
REAL_COLOR = "#4CAF50"
FRAME_SAMPLE_RATE = 30
LAST_CONV_LAYER_NAME = "Conv_1"   # fallback; auto-search implemented if not found
APP_ICON_PATH = "app_logo.png"

# --- MODEL LOADING ---
model = None
MODEL_INPUT_SIZE = None  # (height, width)

def safe_load_model(path):
    global MODEL_INPUT_SIZE, IMG_SIZE
    try:
        m = tf.keras.models.load_model(path)
        # infer expected input size (channels last)
        try:
            inp_shape = m.input_shape  # (None, H, W, C)
            if inp_shape and len(inp_shape) >= 3:
                h = int(inp_shape[1]) if inp_shape[1] is not None else None
                w = int(inp_shape[2]) if inp_shape[2] is not None else None
                if h and w:
                    MODEL_INPUT_SIZE = (h, w)
                    IMG_SIZE = MODEL_INPUT_SIZE
        except Exception:
            # fallback to default IMG_SIZE
            MODEL_INPUT_SIZE = IMG_SIZE
        print(f"✅ Model loaded from {path}. Expected input size: {MODEL_INPUT_SIZE}")
        return m
    except Exception as e:
        print(f"❌ Error loading model from {path}: {e}")
        return None

model = safe_load_model(MODEL_PATH)


# --- PREDICTION helper (use model.predict for compatibility) ---
def predict_image_array(arr):
    """
    arr: numpy array with shape (1, H, W, 3), dtype float32
    Returns probability scalar between 0..1
    """
    if model is None:
        raise RuntimeError("Model not loaded")
    # ensure float32
    arr = arr.astype("float32")
    # use model.predict (safer cross-version)
    preds = model.predict(arr, verbose=0)
    # handle shapes: preds can be (1,1) or (1,) depending on model
    p = float(np.asarray(preds).reshape(-1)[0])
    return p


# --- CORE LOGIC ---
def preprocess(img):
    """
    img: PIL.Image
    returns: numpy array shape (1, H, W, 3) normalized to [0,1]
    """
    # ensure correct input size
    target = IMG_SIZE
    if MODEL_INPUT_SIZE:
        target = MODEL_INPUT_SIZE
    img = img.convert("RGB").resize(target, Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)


def classify(img):
    """
    img: PIL.Image
    returns: (label_str, confidence_percent, time_taken_seconds)
    """
    if model is None:
        return "ERROR", 0.0, 0.0

    arr = preprocess(img)
    start_time = time.time()
    try:
        p = predict_image_array(arr)
    except Exception as e:
        print("Prediction error:", e)
        return "ERROR", 0.0, 0.0
    end_time = time.time()
    t = end_time - start_time

    # interpret
    if p < 0.5:
        label = "FAKE"
        conf = (1 - p) * 100.0
    else:
        label = "REAL"
        conf = p * 100.0
    return label, conf, t


def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    return None


# --- Grad-CAM helper (robust search for last conv layer) ---
def find_last_conv_layer(m):
    # prefer explicit name if present
    if LAST_CONV_LAYER_NAME:
        try:
            return m.get_layer(LAST_CONV_LAYER_NAME)
        except Exception:
            pass
    # otherwise search from the end for a Conv2D layer
    for layer in reversed(m.layers):
        # check by class name
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer
        # also consider depthwise convs in Mobilenet
        if "conv" in layer.name.lower() and len(layer.output_shape) == 4:
            return layer
    return None


def generate_gradcam(img):
    """
    img: PIL.Image
    returns PIL.Image of superimposed heatmap or None on failure
    """
    if model is None:
        return None

    arr = preprocess(img)  # (1,H,W,3)
    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        print("⚠️ No convolutional layer found for Grad-CAM.")
        return None

    try:
        grad_model = tf.keras.models.Model([model.inputs], [last_conv.output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)
            # target is the logit/prob for the positive class (index 0)
            # predictions shape (1,1) or (1,), take scalar
            target = predictions[:, 0]

        grads = tape.gradient(target, conv_outputs)
        # pooled gradients across spatial locations
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # HxWxC

        # multiply each channel by corresponding gradient importance
        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = tf.maximum(heatmap, 0)
        max_val = tf.reduce_max(heatmap)
        if max_val == 0:
            heatmap = tf.zeros_like(heatmap)
        else:
            heatmap /= max_val
        heatmap = heatmap.numpy()

        # prepare overlay
        original_img = img.convert("RGB").resize(SINGLE_DISPLAY_SIZE)
        orig_np = np.array(original_img).astype("float32") / 255.0

        heatmap_resized = cv2.resize(heatmap, SINGLE_DISPLAY_SIZE)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        superimposed = cv2.addWeighted(np.uint8(orig_np * 255), 0.6, heatmap_color, 0.4, 0)
        cam_img = Image.fromarray(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
        return cam_img
    except Exception as e:
        print("Grad-CAM error:", e)
        return None


# --- GUI (kept same as your original beautiful UI) ---
class MediaShieldApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # CustomTKinter setup
        self.title("🛡️ MediaShield Detector")
        self.geometry("1000x750")
        ctk.set_appearance_mode("Dark")
        ctk.set_default_color_theme("dark-blue")

        # icon
        try:
            icon_image = Image.open(APP_ICON_PATH)
            self.iconphoto(True, icon_image)
        except FileNotFoundError:
            print(f"⚠️ Warning: Application icon not found at {APP_ICON_PATH}. Skipping icon setup.")
        except Exception as e:
            print(f"⚠️ Warning: Error loading application icon: {e}. Skipping icon setup.")

        self.current_img = None
        self.history = []

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Title
        self.title_label = ctk.CTkLabel(self,
                                        text="🛡️ MediaShield Detector",
                                        font=ctk.CTkFont(size=30, weight="bold", family="Arial"),
                                        text_color=PRIMARY_COLOR)
        self.title_label.grid(row=0, column=0, padx=20, pady=(30, 10))

        # Tabview
        self.tab_view = ctk.CTkTabview(self, segmented_button_fg_color=PRIMARY_COLOR,
                                      segmented_button_selected_color=PRIMARY_COLOR)
        self.tab_view.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")

        self.detector_tab = self.tab_view.add("🔍 Detector")
        self.detector_tab.grid_columnconfigure(0, weight=1)
        self.detector_tab.grid_rowconfigure(2, weight=1)

        self._setup_detector_tab(self.detector_tab)

        self.history_tab = self.tab_view.add("📜 History")
        self.history_tab.grid_columnconfigure(0, weight=1)
        self.history_tab.grid_rowconfigure(0, weight=1)
        self._setup_history_tab(self.history_tab)

        model_status = "Model Status: Loaded" if model else "Model Status: ERROR"
        self.status_label = ctk.CTkLabel(self, text=model_status, font=ctk.CTkFont(size=10), text_color="gray")
        self.status_label.grid(row=2, column=0, sticky="s", pady=(0, 5))

    def _setup_detector_tab(self, parent):
        result_frame = ctk.CTkFrame(parent, fg_color="transparent")
        result_frame.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        result_frame.grid_columnconfigure((0, 1), weight=1)

        self.result_label = ctk.CTkLabel(result_frame, text="Awaiting Media",
                                         font=ctk.CTkFont(size=40, weight="bold", family="Arial"))
        self.result_label.grid(row=0, column=0, padx=10, pady=5, sticky="e")

        self.confidence_label = ctk.CTkLabel(result_frame, text="Confidence: N/A | Time: N/A",
                                             font=ctk.CTkFont(size=18, weight="normal", family="Arial"), text_color="gray")
        self.confidence_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")

        self.file_path_label = ctk.CTkLabel(result_frame, text="No file loaded",
                                            font=ctk.CTkFont(size=12, weight="normal", family="Arial"), text_color="gray50")
        self.file_path_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(5, 0), sticky="n")

        self.progress_bar = ctk.CTkProgressBar(parent, mode="determinate", height=10, fg_color="#333", progress_color=FAKE_COLOR)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=1, column=0, padx=40, pady=(0, 5), sticky="ew")

        self.display_frame = ctk.CTkFrame(parent, fg_color=('#ffffff', '#2A3439'), corner_radius=10)
        self.display_frame.grid(row=2, column=0, padx=40, pady=(10, 20), sticky="nsew")
        self.display_frame.grid_columnconfigure((0, 1), weight=1)
        self.display_frame.grid_rowconfigure(0, weight=1)

        self.label_orig = ctk.CTkLabel(self.display_frame, text="Original Image", text_color="gray",
                                       font=ctk.CTkFont(size=16, weight="bold"))
        self.label_orig.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.label_cam = ctk.CTkLabel(self.display_frame, text="Visualization (Grad-CAM)", text_color="gray",
                                      font=ctk.CTkFont(size=16, weight="bold"))
        self.label_cam.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        button_frame = ctk.CTkFrame(parent, fg_color="transparent")
        button_frame.grid(row=3, column=0, padx=20, pady=(10, 30))

        self.button_img = ctk.CTkButton(button_frame, text="📸 Load Image (With Grad-CAM)",
                                        command=lambda: self.load_media(mode='image'),
                                        width=250, height=50, fg_color=PRIMARY_COLOR,
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.button_img.pack(side="left", padx=20)

        self.button_vid = ctk.CTkButton(button_frame, text="🎞️ Analyze Full Video",
                                        command=lambda: self.load_media(mode='video'),
                                        width=250, height=50, fg_color=PRIMARY_COLOR,
                                        font=ctk.CTkFont(size=16, weight="bold"))
        self.button_vid.pack(side="left", padx=20)

    def _setup_history_tab(self, parent):
        ctk.CTkLabel(parent, text="Session History",
                     font=ctk.CTkFont(size=24, weight="bold", family="Arial"), text_color=PRIMARY_COLOR).pack(pady=(15, 5))

        self.history_scroll_frame = ctk.CTkScrollableFrame(parent, label_text="Recent Detections",
                                                           label_fg_color=PRIMARY_COLOR,
                                                           label_font=ctk.CTkFont(size=14, weight="bold"))
        self.history_scroll_frame.pack(padx=20, pady=10, fill="both", expand=True)

        self.history_scroll_frame.grid_columnconfigure(0, weight=1)
        self.history_scroll_frame.grid_columnconfigure(1, weight=0)
        self.history_scroll_frame.grid_columnconfigure(2, weight=0)

        self.refresh_history_display()

    def add_to_history(self, filename, result, confidence, time_taken):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history.insert(0, {
            "timestamp": timestamp,
            "filename": filename,
            "result": result,
            "confidence": f"{confidence:.2f}%",
            "time": f"{time_taken:.2f}s",
            "color": FAKE_COLOR if result == "FAKE" or (result == "FAKE!" and "Video" in filename) else REAL_COLOR
        })
        self.refresh_history_display()

    def refresh_history_display(self):
        for widget in self.history_scroll_frame.winfo_children():
            widget.destroy()

        if not self.history:
            ctk.CTkLabel(self.history_scroll_frame, text="No detection history yet.", text_color="gray").grid(row=0, column=0, columnspan=3, pady=20)
            return

        header_font = ctk.CTkFont(size=13, weight="bold")
        ctk.CTkLabel(self.history_scroll_frame, text="Time | File", font=header_font).grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)
        ctk.CTkLabel(self.history_scroll_frame, text="Result", font=header_font).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        ctk.CTkLabel(self.history_scroll_frame, text="Conf | Time", font=header_font).grid(row=0, column=2, sticky="w", padx=(5, 10), pady=5)

        for i, item in enumerate(self.history):
            row_num = i + 1
            bg_color = ("gray80" if i % 2 == 0 else "gray90", "gray15" if i % 2 == 0 else "gray10")

            item_frame = ctk.CTkFrame(self.history_scroll_frame, fg_color=bg_color, corner_radius=5)
            item_frame.grid(row=row_num, column=0, columnspan=3, sticky="ew", pady=1)
            item_frame.grid_columnconfigure(0, weight=1)
            item_frame.grid_columnconfigure(1, weight=0)
            item_frame.grid_columnconfigure(2, weight=0)

            file_text = f"{item['timestamp']} | {item['filename']}"
            ctk.CTkLabel(item_frame, text=file_text, text_color="white", font=ctk.CTkFont(size=12)).grid(row=0, column=0, sticky="w", padx=(10, 5), pady=5)

            ctk.CTkLabel(item_frame, text=item['result'], text_color=item['color'], font=ctk.CTkFont(size=12, weight="bold")).grid(row=0, column=1, sticky="w", padx=5, pady=5)

            metrics_text = f"{item['confidence']} | {item['time']}"
            ctk.CTkLabel(item_frame, text=metrics_text, text_color="gray", font=ctk.CTkFont(size=12)).grid(row=0, column=2, sticky="w", padx=(5, 10), pady=5)

    def analyze_full_video(self, file_path, filename):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            self.update_result("Error", "Could not open video.", FAKE_COLOR)
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames_to_check = sum(1 for i in range(frame_count) if i % FRAME_SAMPLE_RATE == 0)

        self.current_img = None
        self.update_display_to_text("Analyzing Video...", color=PRIMARY_COLOR)
        self.progress_bar.set(0)

        self.button_img.configure(state="disabled")
        self.button_vid.configure(state="disabled")

        fake_count = 0
        real_count = 0
        frame_num = 0
        processed_samples = 0

        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % FRAME_SAMPLE_RATE == 0:
                    progress = processed_samples / total_frames_to_check if total_frames_to_check > 0 else 0
                    self.progress_bar.set(progress)
                    self.confidence_label.configure(text=f"Progress: {processed_samples}/{total_frames_to_check} frames sampled")
                    self.update()

                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    label, _, _ = classify(pil_img)

                    if label == "FAKE":
                        fake_count += 1
                    else:
                        real_count += 1

                    processed_samples += 1

                frame_num += 1

        except Exception as e:
            print(f"An error occurred during video analysis: {e}")
            self.update_result("ERROR", "Analysis failed.", FAKE_COLOR)
            return
        finally:
            cap.release()

        end_time = time.time()
        time_taken = end_time - start_time

        total_predictions = fake_count + real_count
        if total_predictions > 0:
            fake_ratio = fake_count / total_predictions
            final_label = "FAKE!" if fake_ratio >= 0.5 else "REAL"
            final_conf = max(fake_ratio, 1 - fake_ratio) * 100

            color = FAKE_COLOR if final_label == "FAKE!" else REAL_COLOR

            summary_text = (
                f"Video Analysis Complete\n"
                f"-----------------------------\n"
                f"Total Samples: {total_predictions}\n"
                f"Fake Frames: {fake_count}\n"
                f"Real Frames: {real_count}\n"
                f"Time Taken: {time_taken:.2f}s"
            )
            self.update_display_to_text(summary_text, color=color, is_summary=True)
            self.update_result(final_label, f"Overall Confidence: {final_conf:.2f}% | Time: {time_taken:.2f}s", color)

            self.add_to_history(f"[Video] {filename}", final_label, final_conf, time_taken)

        else:
            self.update_result("ERROR", "No frames sampled.", FAKE_COLOR)

        self.progress_bar.set(0)
        self.button_img.configure(state="normal")
        self.button_vid.configure(state="normal")

    def update_result(self, result_text, confidence_text, color):
        self.result_label.configure(text=result_text, text_color=color)
        self.confidence_label.configure(text=confidence_text, text_color="gray")
        self.update()

    def update_display_to_text(self, message, color="gray", is_summary=False):
        if hasattr(self, 'img_tk_orig'):
            del self.img_tk_orig
        if hasattr(self, 'img_tk_cam'):
            del self.img_tk_cam

        self.label_orig.configure(image=None,
                                 text=message,
                                 text_color=color,
                                 font=ctk.CTkFont(size=20 if is_summary else 16, weight="bold"))

        self.label_cam.configure(image=None, text="Visualization (Grad-CAM)", text_color="gray")

        self.label_orig.grid(row=0, column=0, columnspan=2, padx=20, pady=20)
        self.update()

    def load_media(self, mode):
        self.tab_view.set("🔍 Detector")
        if mode == 'image':
            title = "Select Image File"
            filetypes = [("Image Files", "*.png;*.jpg;*.jpeg")]
        else:
            title = "Select Video File"
            filetypes = [("Video Files", "*.mp4;*.avi;*.mov;*.webm")]

        file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
        if not file_path:
            return

        filename = os.path.basename(file_path)
        self.file_path_label.configure(text=f"Loaded: {filename}")

        if mode == 'image':
            self.label_orig.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=10, pady=10)
            self.label_cam.grid(row=0, column=1, columnspan=1, sticky="nsew", padx=10, pady=10)

            self.current_img = Image.open(file_path)
            self.process_image(filename)
        else:
            video_thread = Thread(target=self.analyze_full_video, args=(file_path, filename))
            video_thread.start()

    def process_image(self, filename):
        try:
            self.update_result("Processing...", "Analyzing pixels and generating heatmap...", PRIMARY_COLOR)

            label, conf, t_time = classify(self.current_img)
            cam_img = generate_gradcam(self.current_img)

            display_img = self.current_img.resize(SINGLE_DISPLAY_SIZE)
            self.img_tk_orig = ctk.CTkImage(light_image=display_img, dark_image=display_img, size=SINGLE_DISPLAY_SIZE)

            self.label_orig.configure(image=self.img_tk_orig, text="Original Image")

            if cam_img:
                display_cam = cam_img.resize(SINGLE_DISPLAY_SIZE)
                self.img_tk_cam = ctk.CTkImage(light_image=display_cam, dark_image=display_cam, size=SINGLE_DISPLAY_SIZE)
                self.label_cam.configure(image=self.img_tk_cam, text="Grad-CAM Visualization")
            else:
                self.label_cam.configure(image=None, text="Grad-CAM Failed (Check Model Layer Name)", text_color=FAKE_COLOR)

            color = FAKE_COLOR if label == "FAKE" else REAL_COLOR
            result_text = "FAKE!" if label == "FAKE" else "REAL"

            self.update_result(result_text, f"Confidence: {conf:.2f}% | Time: {t_time:.3f}s", color)

            self.add_to_history(filename, result_text, conf, t_time)

        except Exception as e:
            self.update_result("ERROR", "An error occurred.", FAKE_COLOR)
            print(f"An error occurred during file operation, prediction, or Grad-CAM: {e}")


# --- RUN ---
if __name__ == "__main__":
    try:
        from PIL import Image
        import customtkinter as ctk
        import cv2
        import time
        from datetime import datetime
    except ImportError as e:
        print(f"\n**Missing Dependency: {e.name}**")
        print("Please run: pip install Pillow customtkinter opencv-python tensorflow")
        sys.exit(1)

    app = MediaShieldApp()
    app.mainloop()
