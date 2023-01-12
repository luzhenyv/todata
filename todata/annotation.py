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

from typing import List, Tuple


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

    def to_dict(self):
        return self.__dict__

def array_to_base64(image: np.ndarray)->str:
    pass


def from_yolo_dectection(
        image: np.ndarray,
        config: dict,
        class_ids: np.ndarray,
        boxes: np.ndarray,
        confidences: np.ndarray = None,
        save_data: bool = False,
) -> Annotation:
    """Convert the yolov5 detections to ToData Annotation Format

    Args:
        image: the original BGR image
        config: the project configs
        class_ids: a list of object class ID
        boxes: a list of object bundling boxes, NOT NORMALIZED, FORMAT IS XYXYX
        confidences: a list of object confidence
        save_data: bool, if true, store base64-format image data

    Returns:
        annotation: Annotation, a todata-format annotation
    """
    assert len(class_ids) == len(boxes), "IDs And BOXes have unequal length"
    anno = Annotation(
        image_data = array_to_base64(image) if save_data else None,
        image_md5 = hashlib.md5(image.tobytes()).digest(),
        image_shape = image.shape,
    )
    CLASSES = config.get('CLASSES')

    for index, classid in enumerate(class_ids):
        box = boxes[index]
        confidence = None
        if confidences and len(confidences) == len(classid):
            confidence = confidences[index]

        shape = Shape(
            label = CLASSES[classid],
            shape_type = 'rectangle',
            points = [box[:2], box[2:]],
            flags = {'confidence': confidence} if confidence else None
        )
        anno.add_shape(shape)

    return anno
