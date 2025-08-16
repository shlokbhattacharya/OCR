from versions.ocr_base import BaseDigitDrawGUI
from constants import *
from PIL import Image
import numpy as np
import cv2


class StandardDigitDrawGUI(BaseDigitDrawGUI):
    """Standard implementation"""
    
    def find_digit_bboxes(self, pil_img: Image.Image, min_area: int = 30) -> list:
        """
        Returns list of (x0, y0, x1, y1) bounding boxes for each connected component.
        Filters tiny noise via min_area.
        """
        arr = np.array(pil_img)
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
        return boxes
    
    def group_digits(self, digit_predictions):
        """
        Groups nearby digits using graph-search-based approach that checks all possible pairs.
        """
        if not digit_predictions:
            return []
        
        if len(digit_predictions) == 1:
            return [digit_predictions]
        
        # Create adjacency list for digits that should be grouped
        n = len(digit_predictions)
        graph = [[] for _ in range(n)]
        
        # Check all pairs to see if they should be grouped
        for i in range(n):
            for j in range(i + 1, n):
                digit_i = digit_predictions[i]
                digit_j = digit_predictions[j]
                
                # Extract bounding box coordinates
                i_x0, i_y0, i_x1, i_y1 = digit_i[2]
                j_x0, j_y0, j_x1, j_y1 = digit_j[2]
                
                # Calculate metrics
                i_height = i_y1 - i_y0
                j_height = j_y1 - j_y0
                i_width = i_x1 - i_x0
                j_width = j_x1 - j_x0
                
                i_center_y = (i_y0 + i_y1) / 2
                j_center_y = (j_y0 + j_y1) / 2
                
                # Calculate horizontal gap (can be negative if overlapping)
                if i_x0 < j_x0:  # i is to the left of j
                    horizontal_gap = j_x0 - i_x1
                else:  # j is to the left of i
                    horizontal_gap = i_x0 - j_x1
                
                # Check grouping criteria
                same_line = self._are_on_same_line(i_center_y, j_center_y, i_height, j_height)
                close_horizontally = self._are_close_horizontally(horizontal_gap, i_height, j_height)
                                
                if same_line and close_horizontally:
                    graph[i].append(j)
                    graph[j].append(i)
        
        # Find connected components using DFS
        groups = self._run_dfs(graph, n, digit_predictions)
        
        # Sort each group by center x-coordinate (left to right)
        for group in groups:
            group.sort(key=lambda d: (d[2][0]+d[2][2])/2)
        
        # Sort groups by reading-style (top->bottom, left->right)
        groups.sort(key=lambda g: (round((g[0][2][1]+g[0][2][3])/2, -1), g[0][2][0]))
        
        return groups
    
    def _run_dfs(self, graph, n, digit_predictions):
        visited = [False] * n
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
                        
                        # Add all unvisited neighbors
                        for neighbor in graph[node]:
                            if not visited[neighbor]:
                                stack.append(neighbor)
                
                groups.append(group)
        
        return groups
    
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
        
        final_arr = final_arr.reshape(1, 28, 28, 1)

        return final_arr, new_image
    
    def _are_on_same_line(self, y1, y2, h1, h2):
        """Check if two digits are on the same horizontal line"""
        # Allow some vertical overlap/misalignment based on digit heights
        max_height = max(h1, h2)
        vertical_tolerance = max_height * 0.3  # 30% of the larger digit's height
        return abs(y1 - y2) <= vertical_tolerance

    def _are_close_horizontally(self, gap, h1, h2):
        """Check if two digits are sufficiently close horizontally"""
        avg_height = (h1 + h2) / 2
        
        # Max gap is proportional to digit height
        max_gap = avg_height * 0.4
        
        return gap <= max_gap

    def format_grouped_predictions(self, sorted_groups):
        """
        Convert grouped digit predictions into formatted strings and annotations
        
        Args:
            groups: List of digit groups from group_digits()
        
        Returns:
            List of tuples: (number_string, confidence_avg, overall_bbox)
        """
        formatted_groups = []
        
        for sorted_group in sorted_groups:
            if not sorted_group:
                continue
                            
            # Build number string
            number_str = ''.join(str(digit[0]) for digit in sorted_group)
            
            # Calculate average confidence
            avg_confidence = sum(digit[1] for digit in sorted_group) / len(sorted_group)
            
            bbox_margin = 10

            # Calculate overall bounding box for the group
            all_x0 = [digit[2][0] - bbox_margin for digit in sorted_group]
            all_y0 = [digit[2][1] - bbox_margin for digit in sorted_group]
            all_x1 = [digit[2][2] + bbox_margin for digit in sorted_group]
            all_y1 = [digit[2][3] + bbox_margin for digit in sorted_group]
            
            group_bbox = (min(all_x0), min(all_y0), max(all_x1), max(all_y1))
            
            formatted_groups.append((number_str, avg_confidence, group_bbox))
        
        return formatted_groups

    
    def predict_and_display(self):
        """
        Detects all digits, groups them into numbers, predicts each, and updates display.
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

        # Get individual digit predictions
        digit_preds = []
        for (x0, y0, x1, y1) in boxes:
            crop = self.image.crop((x0, y0, x1, y1))
            arr28, pil28 = self.preprocess_for_mnist(crop)
            if arr28 is None:
                digit_preds.append(("?", 0.0, (x0, y0, x1, y1)))
                continue
            else:
                out = self.model.predict(arr28, verbose=0)
                if out.ndim == 2 and out.shape[1] >= 10:
                    probs = out[0]
                    digit = int(np.argmax(probs))
                    conf = float(probs[digit])
                else:
                    digit = int(np.argmax(out[0]))
                    conf = float(np.max(out[0]))
            digit_preds.append((digit, conf, (x0, y0, x1, y1)))

        # Group digits into numbers
        groups = self.group_digits(digit_preds)
        formatted_groups = self.format_grouped_predictions(groups)

        # Draw annotations for each group
        all_numbers = []
        for number_str, avg_conf, (gx0, gy0, gx1, gy1) in formatted_groups:
            if len(number_str) > 1: 
                # Draw bounding box around the entire number
                rid = self.canvas.create_rectangle(gx0, gy0, gx1, gy1, outline=NUMBER_BOX_COLOR, 
                                                width=3, tags=("front_drawing"))
                self._anno_ids.append(rid)
                
                # Add number label
                label_y = gy0 - 15 if gy0 > 20 else gy1 + 15
                tid = self.canvas.create_text((gx0 + gx1) // 2, label_y, text=number_str, 
                                            fill=NUMBER_BOX_COLOR, font=("Helvetica", 16, "bold"), 
                                            tags=("front_drawing"))
                self._anno_ids.append(tid)
            
            all_numbers.append(number_str)

        seq = []
        for digit, conf, (x0, y0, x1, y1) in digit_preds:
            rid = self.canvas.create_rectangle(x0, y0, x1, y1, outline=DIGIT_BOX_COLOR, width=2, tags=("front_drawing"))
            self._anno_ids.append(rid)
            label = f"{digit}" if isinstance(digit, int) else digit
            label_y = y0 - 10 if y0 > 15 else y1 + 10
            tid = self.canvas.create_text((x0 + x1) // 2, label_y, text=label, fill=DIGIT_BOX_COLOR, font=("Helvetica", 14, "bold"), tags=("front_drawing"))
            self._anno_ids.append(tid)
            seq.append(str(label))

        # Update display
        if hasattr(self, "predict_label"):
            prediction_text = f"Numbers: {', '.join(all_numbers)}" if all_numbers else "Prediction: —"
            self.predict_label.config(text=prediction_text)
        if hasattr(self, "status"):
            self.status.config(text=f"Detected {len(formatted_groups)} number(s) with {len(digit_preds)} total digits.")

        return formatted_groups