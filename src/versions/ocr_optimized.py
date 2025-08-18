from versions.ocr_base import BaseDigitDrawGUI
from constants import *
from PIL import Image
import numpy as np
import cv2

class OptimizedDigitDrawGUI(BaseDigitDrawGUI):
    """Optimized implementation with batch processing"""

    def _update_bbox(self, *coords):
        """Efficiently update bounding box"""
        for i in range(0, len(coords), 2):
            x, y = coords[i], coords[i+1]
            self._min_x = min(self._min_x, x)
            self._min_y = min(self._min_y, y)
            self._max_x = max(self._max_x, x)
            self._max_y = max(self._max_y, y)

    def preprocess_batch(self, crops_and_coords):
        """
        Batch preprocessing of multiple digit crops for efficiency
        
        Args:
            crops_and_coords: List of tuples (crop_image, (x0, y0, x1, y1))
        
        Returns:
            batch_array: numpy array of shape (batch_size, 28, 28, 1)
            valid_indices: list of indices for valid preprocessed images
        """
        if not crops_and_coords:
            return np.array([]), []
        
        batch_arrays = []
        valid_indices = []
        
        for i, (crop, coords) in enumerate(crops_and_coords):
            arr28, _ = self.preprocess_for_mnist(crop)
            if arr28 is not None:
                batch_arrays.append(arr28[0])  # Remove batch dimension
                valid_indices.append(i)
        
        if batch_arrays:
            # Stack all arrays into a single batch
            batch_array = np.stack(batch_arrays, axis=0)
            # Add back the channel dimension if needed
            if batch_array.ndim == 3:
                batch_array = batch_array.reshape(-1, 28, 28, 1)
            return batch_array, valid_indices
        
        return np.array([]), []

    def preprocess_for_mnist(self, crop: Image.Image):
        """Optimized preprocessing with early returns"""
        if crop.size[0] == 0 or crop.size[1] == 0:
            return None, None
        
        crop_w, crop_h = crop.size
        
        # Calculate new dimensions
        if crop_w > crop_h:
            new_w, new_h = 20, max(1, int(round((20.0 * crop_h) / crop_w)))
        else:
            new_h, new_w = 20, max(1, int(round((20.0 * crop_w) / crop_h)))

        # Resize and center
        resized = crop.resize((new_w, new_h), Image.LANCZOS)
        new_image = Image.new("L", (28, 28), color=0)
        upper_left = ((28 - new_w) // 2, (28 - new_h) // 2)
        new_image.paste(resized, upper_left)

        # Convert to numpy for faster processing
        arr = np.array(new_image, dtype=np.float32)
        
        # Skip expensive centroid calculation if sum is too small
        if arr.sum() < 100:  # Threshold to avoid processing noise
            final_arr = arr.reshape(1, 28, 28, 1) / 255.0
            return final_arr, new_image

        # Vectorized centroid calculation
        y_coords, x_coords = np.mgrid[0:28, 0:28]
        total_mass = arr.sum()
        cy = (arr * y_coords).sum() / total_mass
        cx = (arr * x_coords).sum() / total_mass
        
        shift_x = int(np.round(14.0 - cx))
        shift_y = int(np.round(14.0 - cy))
        
        # Apply shifts with bounds checking
        if abs(shift_x) < 14 and abs(shift_y) < 14:  # Prevent excessive shifts
            arr = np.roll(arr, shift_x, axis=1)
            arr = np.roll(arr, shift_y, axis=0)
            
            # Zero out wrapped regions efficiently
            if shift_x > 0:
                arr[:, :shift_x] = 0
            elif shift_x < 0:
                arr[:, shift_x:] = 0
            if shift_y > 0:
                arr[:shift_y, :] = 0
            elif shift_y < 0:
                arr[shift_y:, :] = 0

        # Threshold and normalize
        arr = np.where(arr > THRESHOLD_VALUE, 255, 0)
        final_arr = arr.reshape(1, 28, 28, 1).astype(np.float32) / 255.0
        
        return final_arr, Image.fromarray(arr.astype(np.uint8))

    
    def find_digit_bboxes_optimized(self, pil_img: Image.Image, min_area: int = 30):
        """Optimized bounding box detection with vectorized operations"""
        arr = np.array(pil_img, dtype=np.uint8)
        
        # Use faster thresholding
        bw = (arr > THRESHOLD_VALUE).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return []
        
        # Vectorized bounding rectangle calculation
        boxes = []
        H, W = bw.shape
        
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h >= min_area:
                # Vectorized margin calculation
                margin = max(2, int(0.05 * max(w, h)))
                x0 = max(0, x - margin)
                y0 = max(0, y - margin)
                x1 = min(W, x + w + margin)
                y1 = min(H, y + h + margin)
                boxes.append((x0, y0, x1, y1))
        
        return boxes

    def group_digits_optimized(self, digit_predictions):
        """Optimized grouping with early termination and vectorized calculations"""
        if len(digit_predictions) <= 1:
            return [digit_predictions] if digit_predictions else []
        
        n = len(digit_predictions)
        
        # Pre-calculate all metrics in vectorized form
        coords = np.array([pred[2] for pred in digit_predictions])  # (n, 4)
        heights = coords[:, 3] - coords[:, 1]  # (n,)
        widths = coords[:, 2] - coords[:, 0]   # (n,)
        center_ys = (coords[:, 1] + coords[:, 3]) / 2  # (n,)
        
        # Vectorized distance calculations
        y_diffs = np.abs(center_ys[:, None] - center_ys[None, :])  # (n, n)
        max_heights = np.maximum(heights[:, None], heights[None, :])  # (n, n)
        
        # Same line check (vectorized)
        vertical_tolerance = max_heights * 0.3
        same_line_matrix = y_diffs <= vertical_tolerance
        
        # Horizontal distance check (vectorized)
        x0s = coords[:, 0]
        x1s = coords[:, 2]
        
        # Calculate gaps between all pairs
        gaps = np.maximum(x0s[:, None] - x1s[None, :], x0s[None, :] - x1s[:, None])
        avg_heights = (heights[:, None] + heights[None, :]) / 2
        max_gaps = avg_heights * 0.4
        close_horizontal_matrix = gaps <= max_gaps
        
        # Combine criteria
        adjacency_matrix = same_line_matrix & close_horizontal_matrix
        np.fill_diagonal(adjacency_matrix, False)  # Remove self-connections
        
        # Convert to adjacency list for DFS
        graph = [np.where(adjacency_matrix[i])[0].tolist() for i in range(n)]
        
        # Fast DFS using iterative approach
        visited = np.zeros(n, dtype=bool)
        groups = []
        
        for i in range(n):
            if not visited[i]:
                group = []
                stack = [i]
                
                while stack:
                    node = stack.pop()
                    if not visited[node]:
                        visited[node] = True
                        group.append(digit_predictions[node])
                        stack.extend(neighbor for neighbor in graph[node] if not visited[neighbor])
                
                groups.append(group)
        
        # Sort by center x-coordinate
        for group in groups:
            group.sort(key=lambda d: (d[2][0]+d[2][2])/2) 
        
        groups.sort(key=lambda g: (round((g[0][2][1] + g[0][2][3]) / 2, -1), g[0][2][0]))
        
        return groups
    
    def predict_and_display(self):
        digit_preds, formatted_groups = self.predict_nums()
        if not digit_preds or not formatted_groups:
            self.predict_label.config(text="Prediction: —")
        
        self.display_bboxes(digit_preds, formatted_groups)

    def predict_nums(self):
        """Optimized prediction with batch processing"""
        # Use optimized bounding box detection
        boxes = self.find_digit_bboxes_optimized(self.image)
        if not boxes:
            return None, None

        # Prepare crops for batch processing
        crops_and_coords = []
        for coords in boxes:
            crop = self.image.crop(coords)
            crops_and_coords.append((crop, coords))

        # Batch preprocess all crops
        batch_array, valid_indices = self.preprocess_batch(crops_and_coords)
        
        if batch_array.size == 0:
            self.predict_label.config(text="Prediction: —")
            return None, None

        # Batch prediction (much faster than individual predictions)
        batch_predictions = self.model.predict(batch_array, verbose=0)
        
        # Process predictions
        digit_preds = []
        pred_idx = 0
        
        for i, (crop, coords) in enumerate(crops_and_coords):
            if i in valid_indices:
                probs = batch_predictions[pred_idx]
                digit = int(np.argmax(probs))
                conf = float(probs[digit])
                digit_preds.append((digit, conf, coords))
                pred_idx += 1
            else:
                digit_preds.append(("?", 0.0, coords))

        # Use optimized grouping
        groups = self.group_digits_optimized(digit_preds)
        formatted_groups = self.format_grouped_predictions(groups)

        return digit_preds, formatted_groups
    
    def display_bboxes(self, digit_preds, formatted_groups):
        # Clear previous annotations
        for _id in self._anno_ids:
            try:
                self.canvas.delete(_id)
            except:
                pass
        self._anno_ids = []

        # Draw annotations
        all_numbers = []
        for number_str, avg_conf, (gx0, gy0, gx1, gy1) in formatted_groups:
            if len(number_str) > 1:
                rid = self.canvas.create_rectangle(gx0, gy0, gx1, gy1, outline=NUMBER_BOX_COLOR, 
                                                 width=3, tags=("front_drawing"))
                self._anno_ids.append(rid)
                
                label_y = gy0 - 15 if gy0 > 20 else gy1 + 15
                tid = self.canvas.create_text((gx0 + gx1) // 2, label_y, text=number_str, 
                                            fill=NUMBER_BOX_COLOR, font=("Helvetica", 16, "bold"), 
                                            tags=("front_drawing"))
                self._anno_ids.append(tid)
            
            all_numbers.append(number_str)

        # Draw individual digit boxes
        for digit, conf, (x0, y0, x1, y1) in digit_preds:
            rid = self.canvas.create_rectangle(x0, y0, x1, y1, outline=DIGIT_BOX_COLOR, 
                                             width=2, tags=("front_drawing"))
            self._anno_ids.append(rid)
            
            label = str(digit) if isinstance(digit, int) else digit
            label_y = y0 - 10 if y0 > 15 else y1 + 10
            tid = self.canvas.create_text((x0 + x1) // 2, label_y, text=label, 
                                        fill=DIGIT_BOX_COLOR, font=("Helvetica", 14, "bold"), 
                                        tags=("front_drawing"))
            self._anno_ids.append(tid)

        # Update display
        prediction_text = f"Numbers: {', '.join(all_numbers)}" if all_numbers else "Prediction: —"
        self.predict_label.config(text=prediction_text)
        self.status.config(text=f"Detected {len(formatted_groups)} number(s) with {len(digit_preds)} total digits.")


    def format_grouped_predictions(self, sorted_groups):
        """Same as before, but more efficient"""
        formatted_groups = []
        margin = 10
        
        for sorted_group in sorted_groups:
            if not sorted_group:
                continue
            
            # sorted_group = sorted(group, key=lambda d: d[2][0])
            number_str = ''.join(str(digit[0]) for digit in sorted_group)
            avg_confidence = sum(digit[1] for digit in sorted_group) / len(sorted_group)
            
            # Vectorized bounding box calculation
            coords_array = np.array([digit[2] for digit in sorted_group])
            group_bbox = (
                coords_array[:, 0].min() - margin,
                coords_array[:, 1].min() - margin, 
                coords_array[:, 2].max() + margin,
                coords_array[:, 3].max() + margin
            )
            
            formatted_groups.append((number_str, avg_confidence, group_bbox))
        
        return formatted_groups
