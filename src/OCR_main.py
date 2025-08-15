"""
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
    - Shows prediction in the GUI.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw
import numpy as np
from tensorflow.keras.models import load_model
import cv2

THRESHOLD_VALUE = 50


class DigitDrawGUI:
    def __init__(self, model=None, canvas_size=1024, stroke_width=5):
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

        btn_frame = ttk.Frame(main)
        btn_frame.grid(row=2, column=1, sticky="we", pady=(6, 0))
        btn_frame.columnconfigure((0, 1), weight=1)

        self.clear_btn = ttk.Button(btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.grid(row=0, column=0, sticky="we", padx=(0, 6))

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
        self.predict_and_display()
        # reset last coords
        self._last_x = None
        self._last_y = None

    # ---------- Utilities ----------
    def draw_bounding_box(self):
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
        self._box_id = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#FF0000", width=2)

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

    # ---------- Preprocessing (MNIST style) ----------
    def preprocess_for_mnist(self, crop: Image.Image):
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

            new_image = new_image.point(lambda p: 255 if p > THRESHOLD_VALUE else 0)

        # Normalize to [0,1] float
        final_arr = np.array(new_image).astype(np.float32) / 255.0
        # MNIST typically has white digits on black background; keep that convention.
        final_arr = final_arr.reshape(1, 28, 28, 1)

        return final_arr, new_image

    def find_digit_bboxes(self, pil_img: Image.Image, min_area: int = 30) -> list:
        """
        Returns list of (x0, y0, x1, y1) bounding boxes for each connected component,
        sorted left-to-right. Filters tiny noise via min_area.
        """
        arr = np.array(pil_img)        # shape (H,W), dtype=uint8, 0..255
        _, bw = cv2.threshold(arr, THRESHOLD_VALUE, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        H, W = bw.shape[:2]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                m = max(2, int(0.05 * max(w, h)))
                x0 = max(0, x - m)
                y0 = max(0, y - m)
                x1 = min(W, x + w + m)
                y1 = min(H, y + h + m)
                boxes.append((x0, y0, x1, y1))
        # sort left→right
        boxes.sort(key=lambda b: b[0])
        return boxes

    # # ---------- Prediction ----------
    def predict_and_display(self):
        """
        Detects all digits, predicts each, annotates the canvas, and updates labels.
        """
        # Clear previous annotations
        if hasattr(self, "_anno_ids"):
            for _id in self._anno_ids:
                try:
                    self.canvas.delete(_id)
                except Exception:
                    pass
        self._anno_ids = []

        boxes = self.find_digit_bboxes(self.image)
        if not boxes:
            if hasattr(self, "predict_label"):
                self.predict_label.config(text="Prediction: —")
            if hasattr(self, "status"):
                self.status.config(text="No digits detected.")
            return

        preds = []
        for (x0, y0, x1, y1) in boxes:
            crop = self.image.crop((x0, y0, x1, y1))
            arr28, pil28 = self.preprocess_for_mnist(crop)
            if arr28 is None:
                preds.append(("?", 0.0, (x0, y0, x1, y1)))
                continue
            if self.model is None:
                digit, conf = "?", 0.0
            else:
                out = self.model.predict(arr28)
                if out.ndim == 2 and out.shape[1] >= 10:
                    probs = out[0]
                    digit = int(np.argmax(probs))
                    conf = float(probs[digit])
                else:
                    digit = int(np.argmax(out[0]))
                    conf = float(np.max(out[0]))
            preds.append((digit, conf, (x0, y0, x1, y1)))

        seq = []
        for digit, conf, (x0, y0, x1, y1) in preds:
            rid = self.canvas.create_rectangle(x0, y0, x1, y1, outline="#00FF00", width=2)
            self._anno_ids.append(rid)
            label = f"{digit}" if isinstance(digit, int) else digit
            label_y = y0 - 10 if y0 > 15 else y1 + 10
            tid = self.canvas.create_text((x0 + x1) // 2, label_y, text=label, fill="#00FF00", font=("Helvetica", 14, "bold"))
            self._anno_ids.append(tid)
            seq.append(str(label))

        if hasattr(self, "predict_label"):
            self.predict_label.config(text=f"Prediction: {' '.join(seq)}")
        if hasattr(self, "status"):
            self.status.config(text="Multi-digit prediction complete.")

    def run(self):
        self.root.mainloop()


def main():
    model = load_model('src/model/CNN_model.keras')
    app = DigitDrawGUI(model=model)
    app.run()


if __name__ == "__main__":
    main()
