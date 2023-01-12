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

from .annotation import from_yolo_dectection
from .file import allowed_file

# TODO Design Project YAML format config
yolo_config_demo = {
    'INPUT_SHAPE': [640, 640, 3],
    'SCORE_THRESHOLD': 0.2,
    'CONFIDENCE_THRESHOLD': 0.4,
    'NMS_THRESHOLD': 0.45,
    'CLASSES': ['EOS', 'LYM', 'MAC', 'NEU', 'OTHER'],
    'COLORS': [
        (0, 255, 255),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 0),
        (255, 0, 255),
        (255, 105, 180),
        (255, 69, 0),
    ],
    'LINE_THICKNESS': 2,
}

bp = Blueprint('model', __name__, url_prefix='/model')


# TODO: Consider different file inputs
# TODO: AI model store into USER SESSION
@bp.route('/predict/<filename>', methods=['POST'])
def predict(filename):
    filename = secure_filename(filename)
    assert allowed_file(filename), f'File {filename} is illegal Format'
    file_path = Path(current_app.config['UPLOAD_FOLDER']) / filename
    image = cv2.imread(file_path)  # Read image
    _image = format_yolov5(image)  # wrap original image

    # Model Predict
    net = build_model(current_app.config['MODEL_WEIGHT_PATH'])  # import model
    start_time = time.time()
    predictions = detect(_image, net, yolo_config_demo)
    end_time = time.time()
    current_app.logger.debug(f'Time of detecting is {end_time - start_time}')

    # Unwrap detection
    class_ids, boxes, confidences = unwrap_detection(
        _image, predictions[0], yolo_config_demo,
    )

    annotation = from_yolo_dectection(
        _image, yolo_config_demo, class_ids, boxes, confidences
    )

    # Render Detection
    render_image = render_detection(
        image, yolo_config_demo, class_ids, boxes, confidences
    )

    # Unfinished
    def cv2_to_base64():
        pass

    return jsonify({
        'anno': annotation.to_dict(),
        'render': cv2_to_base64(render_image)
    })


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


def detect(image: np.ndarray, net: Any, config: dict) -> np.ndarray:
    """ Detect objects in the image

    resize to 640x640, normalize to [0, 1] and swap Red
    and Blue channels. by default, OpenCV
    loads colored images as BGR, but yolov5 is RGB.

    Args:
        image: an input normalized RBG image
        net: a onnx format yolov5 model
        config: the project configs

    Returns:
        predictions: the model prediction

    """
    INPUT_HEIGHT, INPUT_WIDTH = config.get('INPUT_SHAPE')[:2]
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


def unwrap_detection(
        image: np.ndarray,
        detection: np.ndarray,
        config: dict,
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
        config: the project configs

    Returns:
        class_ids: a list of object class ID
        confidences: a list of object confidence
        boxes: a list of object bundling boxes, NOT NORMALIZED, FORMAT is XYXY - [left, top, right, bottom]
    """
    class_ids, confidences, boxes = [], [], []

    height, width, _ = image.shape
    INPUT_HEIGHT, INPUT_WIDTH = config.get('INPUT_SHAPE')[:2]

    x_factor = width / INPUT_WIDTH
    y_factor = height / INPUT_HEIGHT

    for row in detection:
        confidence = row[4]
        if confidence < config.get('CONFIDENCE_THRESHOLD'):
            continue

        _, max_value, _, max_index = cv2.minMaxLoc(row[5:])
        if max_value < config.get('SCORE_THRESHOLD'):
            continue

        confidences.append(confidence)
        class_ids.append(max_index[1])
        x, y, w, h = row[:4]
        left = int((x - 0.5 * w) * x_factor)
        top = int((y - 0.5 * h) * y_factor)
        right = int((x + 0.5 * w) * x_factor)
        bottom = int((y + 0.5 * h) * y_factor)
        boxes.append([left, top, right, bottom])

    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        config.get('SCORE_THRESHOLD'),
        config.get('NMS_THRESHOLD'),
    )
    class_ids = np.asarray(class_ids)[indexes]
    confidences = np.asarray(confidences)[indexes]
    boxes = np.asarray(boxes)[indexes]

    return class_ids, boxes, confidences,


def render_detection(
        image: np.ndarray,
        config: dict,
        class_ids: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray,

) -> np.ndarray:
    """Render yolov5 detections

    Args:
        image: the original BGR image
        config: the project configs
        class_ids: a list of object class ID
        boxes: a list of object bundling boxes, NOT NORMALIZED, FORMAT IS XYXYX
        confidences: a list of object confidence

    Returns:
        _image: np.ndarray, a detection rendering image
    """
    assert len(class_ids) == len(boxes) == len(confidences)
    COLORS = config.get('COLORS')
    CLASSES = config.get('CLASSES')
    _image = image.copy()

    for (classid, confidence, box) in zip(class_ids, confidences, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        cv2.rectangle(
            _image,
            box[:2],
            box[2:],
            color,
            config.get('LINE_THICKNESS')
        )  # bounding boxes
        cv2.rectangle(
            _image,
            (box[0], box[1] - 20),
            (box[2], box[1]),
            color,
            -1
        )  # text background
        cv2.putText(
            _image,
            f'{CLASSES[int(classid)]} - {100 * confidence:.1f}%',
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
        )  # Text

    return _image
