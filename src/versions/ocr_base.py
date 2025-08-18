from constants import *
from PIL import Image, ImageDraw

try:
    import tkinter as tk
    from tkinter import ttk
except:
    tk = None
    ttk = None

class BaseDigitDrawGUI:
    """Base class with common functionality for both versions"""
    
    def __init__(self, model, show_ui=True):
        self.model = model
        self.canvas_size = CANVAS_SIZE
        self.stroke_width = STANDARD_STROKE_WIDTH
        self.stroke_color = "white"
        self.show_ui = show_ui
        self._anno_ids = []
        
        if show_ui:
            self.setup_ui()
        self.setup_drawing_state()
            
    def setup_drawing_state(self):
        """Initialize drawing state variables"""
        self._drawn = False
        self._last_x = None
        self._last_y = None
        self._min_x = self.canvas_size
        self._min_y = self.canvas_size
        self._max_x = 0
        self._max_y = 0
        self._box_id = None
        self._strokes = []
        self._temp_points = []
        self._anno_ids = []
        
        # Initialize PIL image
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self._image_history = [self.image.copy()]
        
    def setup_ui_in_container(self, container):
        """Setup UI within a given container"""
        self.main_frame = ttk.Frame(container)
        self.main_frame.pack(fill="both", expand=True)
        
        # Left side controls
        controls_frame = ttk.Frame(self.main_frame)
        controls_frame.grid(row=0, column=0, sticky="nw", padx=(10, 12))
        
        # Top buttons (Eraser and Undo)
        top_btn_frame = ttk.Frame(controls_frame)
        top_btn_frame.pack(fill="x", pady=(10, 0))
        
        self.eraser_btn = ttk.Button(top_btn_frame, text="Eraser", command=self.toggle_eraser)
        self.eraser_btn.pack(fill="x", pady=2)
        
        self.undo_btn = ttk.Button(top_btn_frame, text="Undo Stroke", command=self.undo_stroke)
        self.undo_btn.pack(fill="x", pady=2)
        
        # Spacer between top and bottom buttons
        spacer_frame = ttk.Frame(controls_frame)
        spacer_frame.pack(fill="x", pady=(20, 20))
        
        # Bottom button (Clear)
        bottom_btn_frame = ttk.Frame(controls_frame)
        bottom_btn_frame.pack(fill="x")
        
        self.clear_btn = ttk.Button(bottom_btn_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(fill="x", pady=2)
        
        # Canvas in the center
        self.canvas = tk.Canvas(self.main_frame, width=self.canvas_size, height=self.canvas_size, 
                               bg="black", highlightthickness=2, highlightbackground="#444")
        self.canvas.grid(row=0, column=1, padx=12)
        
        # Bottom prediction label
        prediction_frame = ttk.Frame(container)
        prediction_frame.pack(fill="x", pady=(10, 0))
        
        self.predict_label = ttk.Label(prediction_frame, text="Prediction: —", font=("Helvetica", 14))
        self.predict_label.pack(anchor="w", padx=10)
        
        # Status
        status_frame = ttk.Frame(container)
        status_frame.pack(fill="x", pady=(5, 0))
        
        self.status = ttk.Label(status_frame, text="Draw digits with left mouse button. Performance is being measured!", 
                               relief="sunken", anchor="w")
        self.status.pack(fill="x")
        
        # Bind events
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        
    # Event handlers (shared by both versions)
    def on_button_press(self, event):
        self._last_x, self._last_y = event.x, event.y
        self._drawn = True
        self._update_bbox(event.x, event.y)

    def on_motion(self, event):
        if not self._drawn:
            return
        
        x, y = event.x, event.y
        line_id = self.canvas.create_line(self._last_x, self._last_y, x, y,
                                        fill=self.stroke_color, width=self.stroke_width, 
                                        capstyle=tk.ROUND, smooth=True)
        self._temp_points.append(line_id)
        
        fill_color = 255 if self.stroke_color == "white" else 0
        self.draw.line([self._last_x, self._last_y, x, y], fill=fill_color, width=self.stroke_width)
        self.canvas.tag_raise("front_drawing", 'all')
        
        self._update_bbox(x, y, self._last_x, self._last_y)
        self._last_x, self._last_y = x, y

    def _update_bbox(self, *coords):
        """Efficiently update bounding box"""
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i+1]
            self._min_x = min(self._min_x, x)
            self._min_y = min(self._min_y, y)
            self._max_x = max(self._max_x, x)
            self._max_y = max(self._max_y, y)

    def on_button_release(self, event):
        if not self._drawn:
            return
        
        self._strokes.append(self._temp_points[:])
        self._temp_points.clear()
        self.predict_and_display()
        self._image_history.append(self.image.copy())
        self._last_x = None
        self._last_y = None

    def clear_canvas(self):
        self.canvas.delete("all")
        self.setup_drawing_state()
        self.predict_label.config(text="Prediction: —")

        # Start on pen mode
        if self.eraser_btn["text"] == "Pen":
            self.toggle_eraser()

    def toggle_eraser(self):
        if self.eraser_btn["text"] == "Eraser":
            self.stroke_color = "black"
            self.stroke_width = ERASER_MULTIPLIER * STANDARD_STROKE_WIDTH
            self.eraser_btn.configure(text="Pen")
        else:
            self.stroke_color = "white"
            self.stroke_width = STANDARD_STROKE_WIDTH
            self.eraser_btn.configure(text="Eraser")

    def undo_stroke(self):
        if len(self._image_history) <= 1 or not self._strokes:
            return
        
        self._image_history.pop()
        last_image = self._image_history[-1].copy()
        self.image = last_image
        self.draw = ImageDraw.Draw(self.image)
        
        last_stroke = self._strokes.pop()
        self.canvas.delete(*last_stroke)
        self.predict_and_display()