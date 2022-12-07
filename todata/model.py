import cv2
import numpy as np
import time

from flask import (
    abort, Blueprint, current_app, flash, g, jsonify,
    redirect, render_template, request, url_for
)
from typing import Any, Tuple
from pathlib import Path
from werkzeug.utils import secure_filename

from .file import allowed_file

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.4
CLASSES = ['EOS', 'LYM', 'MAC', 'NEU', 'OTHER']
COLORS = [
    (0, 255, 255),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 0),
    (255, 0, 255),
    (255, 105, 180),
    (255, 69, 0),
]

bp = Blueprint('model', __name__, url_prefix='/model')

# TODO: Consider different file inputs
# TODO: AI model store into USER SESSION
@bp.route('/predict/<filename>', methods=['POST'])
def predict(filename):
    filename = secure_filename(filename)
    assert allowed_file(filename), f'File {filename} is illegal Format'
    file_path = Path(current_app.config['UPLOAD_FOLDER']) / filename
    image = cv2.imread(file_path)  # Read image
    _image = format_yolov5(image)  # copy original image

    # Model Predict
    net = build_model(current_app.config['MODEL_WEIGHT_PATH'])  # import model
    start_time = time.time()
    predictions = detect(_image, net)
    end_time = time.time()
    current_app.logger.debug(f'Time of detecting is {end_time - start_time}')

    # Unwrap detection
    class_ids, confidences, boxes = unwrap_detection(
        _image, predictions[0]
    )






def load_capture() -> np.ndarray:
    """Load video frames"""
    capture = cv2.VideoCapture("sample.mp4")
    return capture

def build_model(weight: str, is_cuda: bool = False) -> Any:
    """Loading the YOLOv5 model

    Args:
        weight: model path or url
        is_cuda: whether use cuda or not
    Returns:
        net: a opencv dnn net supporting onnx
    """
    net = cv2.dnn.readNet(weight)
    if is_cuda:
        print("Attempty to use CUDA")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        print("Running on CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def detect(image: np.ndarray, net: Any) -> np.ndarray:
    """ Detect objects in the image

    resize to 640x640, normalize to [0, 1] and swap Red
    and Blue channels. by default, OpenCV
    loads colored images as BGR, but yolov5 is RGB.

    Args:
        image: an input normalized RBG image
        net: a onnx format yolov5 model

    Returns:
        predictions: the model prediction

    """
    blob = cv2.dnn.blobFromImage(
        image,
        1 / 255.0,
        (INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False
    )
    net.setInput(blob)
    predictions = net.forward()
    return predictions


def unwrap_detection(
        image: np.ndarray,
        detection: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """unwrap the yolov5 detections

    YOLOv5's prediction is an an array with 25,200 positions
    where each position is a 85-length 1D array. Each 1D array
    holds the data of one detection. The 4 first positions of
    this array are the xywh coordinates of the bound box rectangle.
    The 5th position is the confidence level of that
    detection. The 6th up to 85th elements are the scores
    of each class (assuming an 80-class coco dataset )

    Args:
        image: the original BGR image
        detection: the yolov5 model predictions,

    Returns:
        class_ids: a list of object class ID
        confidences: a list of object confidence
        boxes: a list of object bundling boxes, NOT NORMALIZED
    """
    class_ids, confidences, boxes = [], [], []

    height, width, _ = image.shape

    x_factor = width / INPUT_WIDTH
    y_factor = height / INPUT_HEIGHT

    for row in detection:
        confidence = row[4]
        if confidence < 0.4:
            continue

        _, max_value, _, max_index = cv2.minMaxLoc(row[5:])
        if max_value < 0.25:
            continue

        confidences.append(confidence)
        class_ids.append(max_index[1])
        x, y, w, h = row[:4]
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        width = int(w * x_factor)
        height = int(h * y_factor)
        boxes.append([left, top, width, height])

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)
    class_ids = np.asarray(class_ids)[indexes]
    confidences = np.asarray(confidences)[indexes]
    boxes = np.asarray(boxes)[indexes]

    return class_ids, confidences, boxes


def format_yolov5(image: np.ndarray) -> np.ndarray:
    """Covert a BGR image to the yolov5 format.

    first, put the image or the video frame in a square big enough;.

    Args:
        image: A BGR [0, 255] OpenCV image.
    Returns:
         resized: a RGB [0, 1] yolov5 format image
    """
    height, width, _ = image.shape
    _max = max(height, width)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:height, 0:width] = image  # put into a square
    return resized




