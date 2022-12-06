"""The content of a image annotation

The 'Annotation' Class stores the content of a image's annotation, or label.
The 'Shape' Class object records the region of interest (ROI) info.

Typical usage example:
    anno = Annotation
"""
import cv2
import json
import hashlib
import numpy as np

from typing import List


class Shape(object):
    """ Contents of a ROI

    Attributes:
        label: a string indicating the ROI class name
        group_id: a string distinguishing same-class but different ROI
        shape_type: a string in ['rectangle','Polygon','circle','line','point'], indicating the ROI shape
        points: a list of ROI vertex coordinates, like [[x1,y1],[x2,y2]], NOT NORMALIZED
        flags: a dict records necessary key-value mapping

    """
    def __init__(
            self,
            label: str,
            group_id: str = None,
            shape_type: str = "polygon",
            points: list = None,
            flags: dict = None
    ):
        self.label = label
        self.group_id = group_id
        self.shape_type = shape_type
        self.points = points if points else []
        self.flags = flags if flags else []


class Annotation(object):
    """Contents of an image's annotation, or label.

    Attributes:
        version: A string declares the version of label module.
        label: A string indicating the image class name.
        flags: A dict recording necessary key-value.
        shapes: A list of Shape objects, including all ROI info in the images.
        image_path: A string of file path to the the annotation object' image.
        image_data: A base64 string of image data.
        image_md5: A string of MD5 used to check
        image_shape: A list indicating the image shape, the format is [H, W, C, ...].
    """

    def __init__(
            self,
            version: str = '1.0.0',
            label: str = None,
            flags: dict = None,
            shapes: list = None,
            image_path: str = None,
            image_data: str = None,
            image_md5: str = None,
            image_shape: list = None,
    ):
        self.version = version
        self.label = label
        self.flags = flags if flags else {}
        self.shapes = shapes if shapes else []
        self.image_path = image_path
        self.image_data = image_data
        self.image_md5 = image_md5
        self.image_shape = image_shape

    def add_shape(self, shape: Shape):
        if isinstance(shape, Shape):
            self.shapes.append(shape)

def array_to_base64(image: np.ndarray)->str:
    pass

def from_yolo_dectection(
        image: np.ndarray,
        detection: np.ndarray,
        input_shape: list,
        save_data: bool = False,
        save_confidence: bool = True,
) -> Annotation:
    """Convert the yolov5 detections to ToData Annotation Format

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
            input_shape: a list indicating yolo model inputs shape, like [H, W, C,...] excluding batch size
            save_data: bool, if true, store base64-format image data
            save_confidence: bool, if true, store the ROI confidence

        Returns:
            annotation: Annotation, a todata-format annotation
        """
    anno = Annotation(
        image_data = array_to_base64(image) if save_data else None,
        image_md5 = hashlib.md5(image.tobytes()).digest(),
        image_shape = image.shape,
    )

    height, width, _ = anno.image_shape
    INPUT_HEIGHT, INPUT_WIDTH = input_shape[:2]
    x_factor = width / INPUT_WIDTH
    y_factor = height / INPUT_HEIGHT

    for row in detection:
        confidence = row[4]
        if confidence < 0.4:
            continue
        _, max_value, _, max_index = cv2.minMaxLoc(row[5:])
        if max_value < 0.25:
            continue

        x, y, w, h = row[:4]




    return anno
