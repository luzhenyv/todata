"""The content of a image annotation

The 'Annotation' Class stores the content of a image's annotation, or label.
The 'Shape' Class object records the region of interest (ROI) info.

Typical usage example:
    anno = Annotation
"""
import json

from typing import List


class Shape(object):
    """ Contents of a ROI

    Attributes:
        label: a string indicating the ROI class name
        group_id: a string distinguishing same-class but different ROI
        shape_type: a string in ['rectangle','Polygon','circle','line','point'], indicating the ROI shape
        points: a list of ROI vertex coordinates, like [[x1,y1],[x2,y2]]
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
            version: str,
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
        self.image_shape = image_shape

    def add_shape(self, shape: Shape):
        if isinstance(shape, Shape):
            self.shapes.append(shape)