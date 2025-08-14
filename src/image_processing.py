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


def run_ocr(image, regions):
    """
    Run Tesseract OCR on each region and only keep those predicted as single digits.
    Returns filtered list of (x, y, w, h).
    """
    filtered = []
    for (x, y, w, h) in regions:
        roi = image[y:y+h, x:x+w]
        # OCR config: only digits
        config = '--psm 7'
        prediction = pytesseract.image_to_string(roi, config=config).strip()
        print(f"text: {prediction}\n")
        filtered.append((x, y, w, h, prediction))
    return filtered


def detect_and_prepare_digits(img):
    """
    Full pipeline: load, color filter, text-region detect, OCR.
    Returns list of (x, y, prediction).
    """
    # Load image
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
    return run_ocr(img, text_regions)
    


if __name__ == '__main__':
    # Example usage:
    # from tensorflow.keras.models import load_model
    # image_path = '/Users/shlokbhattacharya/Downloads/8629.jpeg'
    image_path = '/Users/shlokbhattacharya/Desktop/OCR_Project/opencv-text-detection/images/lebron_james.jpg'
    img = cv2.imread(image_path)
    model = load_model('src/model/CNN_model.keras')
    digits = detect_and_prepare_digits(img)
    # digits = detect_and_prepare_digits('/Users/shlokbhattacharya/Desktop/OCR_Project/opencv-text-detection/images/lebron_james.jpg')

    config = '--psm 7'
    prediction = pytesseract.image_to_string(img, config=config).strip()

    print(digits)
    print(prediction)
