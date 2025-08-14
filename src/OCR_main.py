"""
mnist_draw_gui.py

Usage:
    python mnist_draw_gui.py --model path/to/your/keras_model.h5

Or from another script:
    from tensorflow.keras.models import load_model
    model = load_model("...")   # or however you already have it loaded
    app = DigitDrawGUI(model=model)
    app.run()

Requirements:
    - Python 3.8+
    - Pillow
    - numpy
    - tensorflow (or tensorflow-cpu) if you want to actually run a Keras model

What it does:
    - Hold left mouse button to paint white on a black canvas.
    - On mouse release it draws a bounding box and automatically:
        * crops the drawn area,
        * adds padding, resizes to 20x20 while preserving aspect ratio,
        * centers the glyph in a 28x28 image using center-of-mass,
        * normalizes pixel values to [0,1] and sends (1,28,28,1) to model.predict
    - Shows prediction and confidence in the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk, ImageChops
import numpy as np
import os
from tensorflow.keras.models import load_model


class DigitDrawGUI:
    def __init__(self, model=None, canvas_size=280, stroke_width=15):
        """
        model: either a Keras model object with model.predict or None.
        canvas_size: width/height of drawing canvas in pixels.
        stroke_width: thickness of the drawing stroke in pixels.
        """
        self.model = model
        self.canvas_size = canvas_size
        self.stroke_width = stroke_width

        # Tkinter root
        self.root = tk.Tk()
        self.root.title("OCR Project")

        # Top frame for canvas + controls
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Canvas (black)
        self.canvas = tk.Canvas(main, width=canvas_size, height=canvas_size, bg="black", highlightthickness=2,
                                highlightbackground="#444")
        self.canvas.grid(row=0, column=0, rowspan=4, padx=(0, 12))

        # A PIL image that mirrors the canvas so we can crop/save/process
        self.image = Image.new("L", (canvas_size, canvas_size), color=0)  # 'L' mode, black background
        self.draw = ImageDraw.Draw(self.image)

        # Controls
        self.predict_label = ttk.Label(main, text="Prediction: —", font=("Helvetica", 14))
        self.predict_label.grid(row=0, column=1, sticky="w")

        self.conf_label = ttk.Label(main, text="Confidence: —", font=("Helvetica", 10))
        self.conf_label.grid(row=1, column=1, sticky="w", pady=(2, 10))

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=1, sticky="we", pady=(6, 0))
        btn_frame.columnconfigure((0, 1), weight=1)

        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0, sticky="we", padx=(0, 6))

        self.predict_btn = ttk.Button(btn_frame, text="Predict", command=self.manual_predict)
        self.predict_btn.grid(row=0, column=1, sticky="we")

        self.save_btn = ttk.Button(main, text="Save 28x28 (debug)", command=self.save_processed_debug)
        self.save_btn.grid(row=3, column=1, sticky="we", pady=(10, 0))

        # Status bar
        self.status = ttk.Label(self.root, text="Draw digit by holding left mouse button. Release to predict.", relief="sunken",
                                anchor="w")
        self.status.grid(row=1, column=0, sticky="we")

        # Event handling for drawing
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Track drawing state
        self._drawn = False
        self._last_x = None
        self._last_y = None
        self._min_x = canvas_size
        self._min_y = canvas_size
        self._max_x = 0
        self._max_y = 0
        self._box_id = None

        # Keep a small track of strokes so we can compute bbox robustly
        self._points = []

        # Keep a preview image (tkinter) for the processed 28x28 to show debug (optional)
        self.preview_win = None

    # ---------- Drawing event handlers ----------
    def on_button_press(self, event):
        self._last_x, self._last_y = event.x, event.y
        self._drawn = True
        self._points.append((event.x, event.y))
        # update bbox
        self._min_x = min(self._min_x, event.x)
        self._min_y = min(self._min_y, event.y)
        self._max_x = max(self._max_x, event.x)
        self._max_y = max(self._max_y, event.y)

    def on_motion(self, event):
        if not self._drawn:
            return
        x, y = event.x, event.y
        # draw line on canvas
        self.canvas.create_line(self._last_x, self._last_y, x, y,
                                fill="white", width=self.stroke_width, capstyle=tk.ROUND, smooth=True)
        # draw on PIL image (white)
        self.draw.line([self._last_x, self._last_y, x, y], fill=255, width=self.stroke_width)
        # update bbox
        self._min_x = min(self._min_x, x, self._last_x)
        self._min_y = min(self._min_y, y, self._last_y)
        self._max_x = max(self._max_x, x, self._last_x)
        self._max_y = max(self._max_y, y, self._last_y)
        self._last_x, self._last_y = x, y
        self._points.append((x, y))

    def on_button_release(self, event):
        if not self._drawn:
            return
        # finalize last point
        self._points.append((event.x, event.y))
        # draw bounding box with some small margin
        self.draw_bounding_box()
        # automatically run prediction
        self._predict_and_display()
        # reset last coords
        self._last_x = None
        self._last_y = None

    # ---------- Utilities ----------
    def draw_bounding_box(self):
        # remove previous box if any
        if self._box_id:
            self.canvas.delete(self._box_id)
            self._box_id = None

        if self._min_x >= self._max_x or self._min_y >= self._max_y:
            return

        margin = int(max(2, 0.05 * max(self._max_x - self._min_x, self._max_y - self._min_y)))
        x0 = max(0, self._min_x - margin)
        y0 = max(0, self._min_y - margin)
        x1 = min(self.canvas_size, self._max_x + margin)
        y1 = min(self.canvas_size, self._max_y + margin)

        self._box_coords = (x0, y0, x1, y1)
        self._box_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00FF00", width=2)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self._drawn = False
        self._last_x = None
        self._last_y = None
        self._min_x = self.canvas_size
        self._min_y = self.canvas_size
        self._max_x = 0
        self._max_y = 0
        self._points.clear()
        self._box_coords = None
        self._box_id = None
        self.predict_label.config(text="Prediction: —")
        self.conf_label.config(text="Confidence: —")

    # ---------- Preprocessing (MNIST style) ----------
    def preprocess_for_mnist(self, img, box_coords=None):
        """
        Input:
            img: PIL.Image in 'L' mode (0 black - 255 white), same size as canvas
            box_coords: (x0,y0,x1,y1) optional; if None, compute from non-zero pixels
        Returns:
            processed: numpy array shape (1,28,28,1) dtype float32 normalized 0..1
            pil28: the 28x28 PIL image (L) for debug/save
        Steps:
            - crop to bounding box (with padding)
            - resize keeping aspect ratio so that largest side is 20
            - paste centered into 28x28
            - shift to center-of-mass at (14,14)
            - normalize to 0..1
        """
        if box_coords is None:
            arr = np.array(img)
            ys, xs = np.where(arr > 10)
            if len(xs) == 0:
                # nothing drawn
                return None, None
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
        else:
            x0, y0, x1, y1 = box_coords

        # add extra padding (10-20% of max dimension)
        w = x1 - x0
        h = y1 - y0
        pad = int(max(w, h) * 0.18) + 2
        x0 = max(0, x0 - pad)
        y0 = max(0, y0 - pad)
        x1 = min(self.canvas_size, x1 + pad)
        y1 = min(self.canvas_size, y1 + pad)

        crop = img.crop((x0, y0, x1, y1))

        # Resize to fit in 20x20 box while preserving aspect ratio
        crop_w, crop_h = crop.size
        if crop_w > crop_h:
            new_w = 20
            new_h = max(1, int(round((20.0 * crop_h) / crop_w)))
        else:
            new_h = 20
            new_w = max(1, int(round((20.0 * crop_w) / crop_h)))

        resized = crop.resize((new_w, new_h), Image.LANCZOS)

        # Create a 28x28 black image and paste resized into center
        new_image = Image.new("L", (28, 28), color=0)
        upper_left = ((28 - new_w) // 2, (28 - new_h) // 2)
        new_image.paste(resized, upper_left)

        # Shift centroid to center (14,14)
        arr = np.array(new_image).astype(np.float32)
        # If all zeros (shouldn't happen), skip shift
        if arr.sum() > 0:
            cy = (arr * np.arange(arr.shape[0])[:, None]).sum() / arr.sum()
            cx = (arr * np.arange(arr.shape[1])[None, :]).sum() / arr.sum()
            shift_x = int(np.round(arr.shape[1] / 2.0 - cx))
            shift_y = int(np.round(arr.shape[0] / 2.0 - cy))
            # roll (wraps around) then zero the wrapped in pixels (we'll shift by small amounts so it's okay)
            arr = np.roll(arr, shift_x, axis=1)
            arr = np.roll(arr, shift_y, axis=0)

            # When rolling creates wrapped values, set them to zero:
            if shift_x > 0:
                arr[:, :shift_x] = 0
            elif shift_x < 0:
                arr[:, shift_x:] = 0
            if shift_y > 0:
                arr[:shift_y, :] = 0
            elif shift_y < 0:
                arr[shift_y:, :] = 0

            new_image = Image.fromarray(arr.astype(np.uint8))

        # Normalize to [0,1] float
        final_arr = np.array(new_image).astype(np.float32) / 255.0
        # MNIST typically has white digits on black background; keep that convention.
        final_arr = final_arr.reshape(1, 28, 28, 1)

        return final_arr, new_image

    # ---------- Prediction ----------
    def _predict_and_display(self):
        # If no drawing, do nothing
        if not self._drawn or len(self._points) == 0:
            self.status.config(text="Nothing drawn to predict.")
            return

        array28, pil28 = self.preprocess_for_mnist(self.image, getattr(self, "_box_coords", None))
        if array28 is None:
            self.status.config(text="Unable to process drawing.")
            return

        # Show small preview window with 28x28 for debug
        self.show_preview(pil28)

        if self.model is None:
            self.status.config(text="No model loaded; preprocessing shown.")
            self.predict_label.config(text="Prediction: (no model)")
            self.conf_label.config(text="Confidence: —")
            return

        # Model expects shape (1,28,28,1)
        try:
            preds = self.model.predict(array28)
        except Exception as e:
            messagebox.showerror("Predict error", f"Model prediction failed:\n{e}")
            return

        # handle different output shapes
        if preds.ndim == 2 and preds.shape[1] >= 10:
            probs = preds[0]
            digit = int(np.argmax(probs))
            conf = float(probs[digit])
        else:
            # fallback: if model returns a single scalar or categoricals
            digit = int(np.argmax(preds[0]))
            probs = None
            conf = None
            if hasattr(preds[0], "__len__") and len(preds[0]) > 0:
                conf = float(max(preds[0]))
        self.predict_label.config(text=f"Prediction: {digit}")
        if conf is not None:
            self.conf_label.config(text=f"Confidence: {conf:.3f}")
        else:
            self.conf_label.config(text=f"Confidence: —")
        self.status.config(text="Prediction complete.")

    def manual_predict(self):
        # Force prediction (useful if model not auto-run)
        self._predict_and_display()

    # ---------- Debug/save helpers ----------
    def save_processed_debug(self):
        array28, pil28 = self.preprocess_for_mnist(self.image, getattr(self, "_box_coords", None))
        if array28 is None or pil28 is None:
            messagebox.showinfo("Save", "Nothing to save (no drawing).")
            return
        outpath = "mnist_preview_28x28.png"
        pil28.save(outpath)
        messagebox.showinfo("Saved", f"Saved processed 28x28 image to: {os.path.abspath(outpath)}")

    def show_preview(self, pil_img_28):
        # Show (or update) a small preview window for the 28x28 processed image (scaled up to view)
        scaled = pil_img_28.resize((140, 140), Image.NEAREST)
        if self.preview_win is None or not tk.Toplevel.winfo_exists(self.preview_win):
            self.preview_win = tk.Toplevel(self.root)
            self.preview_win.title("28x28 preview")
            label = ttk.Label(self.preview_win)
            label.pack(padx=6, pady=6)
            self.preview_label = label
        imgtk = ImageTk.PhotoImage(scaled)
        # keep a reference to avoid garbage collection
        self.preview_label.imgtk = imgtk
        self.preview_label.configure(image=imgtk)

    def run(self):
        self.root.mainloop()


def main():
    model = load_model('src/model/CNN_model.keras')
    app = DigitDrawGUI(model=model, canvas_size=280, stroke_width=15)
    app.run()


if __name__ == "__main__":
    main()
