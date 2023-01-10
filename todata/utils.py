import os
import uuid

from urllib.parse import urlparse, urljoin

import PIL
from PIL import Image
from flask import current_app, request, url_for, redirect, flash


def rename_image(old_filename):
    ext = os.path.splitext(old_filename)[1]
    new_filename = uuid.uuid4().hex + ext
    return new_filename


def resize_image(image, filename, base_width):
    filename, ext = os.path.splitext(filename)
    img = Image.open(image)
    if img.size[0] <= base_width:
        return filename + ext
    w_percent = base_width / float(img.size[0])
    h_size = int(img.size[1] * w_percent)
    img = img.resize((base_width, h_size), PIL.Image.ANTIALIAS)

    filename += current_app.config['TODATA_PHOTO_SUFFIX'][base_width] + ext
    img.save(os.path.join(current_app.config['TODATA_UPLOAD_PATH'], filename), optimize=True, quality=85)
    return filename


