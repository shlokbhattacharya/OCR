import cv2
import numpy as np
import pytesseract
# import tensorflow as tf
from scipy.ndimage import center_of_mass
from tensorflow.keras.models import load_model

# ----- Configuration -----
# Path to the pre-trained EAST text detector model (download separately)
EAST_MODEL_PATH = 'src/model/frozen_east_text_detection.pb'
# Confidence threshold for EAST detector
EAST_CONF_THRESHOLD = 0.5
# Non-maximum suppression threshold
EAST_NMS_THRESHOLD = 0.4
# HSV color filter range for digit color (example: white digits on dark background)
HSV_LOWER = np.array([0, 0, 200])
HSV_UPPER = np.array([180, 30, 255])

# ----- Utility Functions -----
def color_filter(img_bgr, lower=HSV_LOWER, upper=HSV_UPPER):
    """
    Mask the image to only keep regions within a specified HSV color range.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    return mask


def detect_text_regions_east(image, net, width=320, height=320):
    """
    Use the EAST text detector to get bounding boxes of text regions.
    Requires image dimensions to be multiples of 32.
    Returns a list of rectangles (x, y, w, h).
    """
    orig_h, orig_w = image.shape[:2]
    rW = orig_w / float(width)
    rH = orig_h / float(height)

    # Resize and prepare blob
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(['feature_fusion/Conv_7/Sigmoid',
                                      'feature_fusion/concat_3'])

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over rows and columns to extract potential boxes
    for y in range(numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(numCols):
            if scoresData[x] < EAST_CONF_THRESHOLD:
                continue
            # Compute offset
            offsetX, offsetY = x * 4.0, y * 4.0
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # Apply non-max suppression
    boxes = cv2.dnn.NMSBoxes(rects, confidences, EAST_CONF_THRESHOLD, EAST_NMS_THRESHOLD)
    results = []
    if len(boxes) > 0:
        for i in boxes.flatten():
            (startX, startY, endX, endY) = rects[i]
            # scale coords back
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            results.append((startX, startY, endX - startX, endY - startY))
    return results


def ocr_filter_regions(image, regions):
    """
    Run Tesseract OCR on each region and only keep those predicted as single digits.
    Returns filtered list of (x, y, w, h).
    """
    filtered = []
    for (x, y, w, h) in regions:
        roi = image[y:y+h, x:x+w]
        # OCR config: only digits
        config = '--psm 10 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(roi, config=config).strip()
        print(f"text: {text}\n")
        if len(text) == 1 and text.isdigit():
            filtered.append((x, y, w, h))
    return filtered


def contour_proposals(binary_img, min_size=20, max_size=300):
    """
    Find contours in a binary image and return bounding boxes within size thresholds.
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if min_size < w < max_size and min_size < h < max_size:
            boxes.append((x, y, w, h))
    return boxes


def process_roi_to_mnist(roi):
    """
    Convert an ROI to a 28x28 grayscale image centered like MNIST.
    """
    # Resize to 20x20
    roi = cv2.resize(roi, (20, 20), interpolation=cv2.INTER_AREA)
    # Pad to 28x28
    padded = np.pad(roi, ((4, 4), (4, 4)), mode='constant', constant_values=0)
    # Center mass
    cy, cx = center_of_mass(padded)
    shift_x = int(np.round(14 - cx))
    shift_y = int(np.round(14 - cy))
    M = np.array([[1, 0, shift_x], [0, 1, shift_y]], dtype=np.float32)
    centered = cv2.warpAffine(padded, M, (28, 28))
    return centered


def detect_and_prepare_digits(image_path, model):
    """
    Full pipeline: load, color filter, text-region detect, OCR filter, contours,
    prepare each ROI, then predict with the model.
    Returns list of (x, y, prediction).
    """
    # Load image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Color filtering
    mask = color_filter(img)
    # Combine color mask + adaptive threshold
    adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY_INV, 15, 10)
    combined = cv2.bitwise_or(mask, adapt)
    
    # 2. EAST text regions
    net = cv2.dnn.readNet(EAST_MODEL_PATH)
    text_regions = detect_text_regions_east(img, net)

    # 3. OCR filter
    ocr_regions = ocr_filter_regions(gray, text_regions)

    # 4. Contour proposals on combined mask
    contour_regions = contour_proposals(combined)

    # Merge proposals
    all_regions = ocr_regions + contour_regions
    # Optionally: apply non-max suppression on all_regions

    results = []
    for (x, y, w, h) in all_regions:
        roi = combined[y:y+h, x:x+w]
        digit_img = process_roi_to_mnist(roi)
        norm = digit_img.astype('float32') / 255.0
        inp = norm.reshape(1, 28, 28, 1)
        pred = np.argmax(model.predict(inp), axis=1)[0]
        results.append((x, y, pred))

    return results
    


if __name__ == '__main__':
    # Example usage:
    # from tensorflow.keras.models import load_model
    model = load_model('src/model/CNN_model.keras')
    # digits = detect_and_prepare_digits('/Users/shlokbhattacharya/Downloads/IMG_8629.jpeg', model)
    digits = detect_and_prepare_digits('/Users/shlokbhattacharya/Desktop/OCR_Project/opencv-text-detection/images/sign.jpg', model)
    print(digits)
    # pass
