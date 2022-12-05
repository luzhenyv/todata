import os
import json

from flask import (
    abort, Blueprint, current_app, flash, g, jsonify,
    redirect, render_template, request, url_for
)
from pathlib import Path
from werkzeug.utils import secure_filename

bp = Blueprint('label', __name__, url_prefix='/label')


# TODO: Check the JSON Format labels
# TODO: Label Version Control
@bp.route('/upload', methods=('GET', 'POST'))
def upload():
    current_app.logger.info('enter label upload')
    if request.method == 'POST':
        # check if the post request has the file part
        if request.files and 'file' in request.files:
            # 文件形式的标注
            label = request.files['file']
            if label and _allowed_file(label.filename):
                filename = secure_filename(label.filename)
                label_path = Path(current_app.config['UPLOAD_FOLDER']) / filename
                label.save(label_path)
                current_app.logger.debug(
                    f'file save path is {label_path}'
                )
                return jsonify({'label_path': label_path})

        current_app.logger.debug(request.json)
        if request.json:
            # JSON 格式的标注
            label = request.json
            if label:
                image_path = Path(label['image_path'])
                label_path = Path(current_app.config['UPLOAD_FOLDER']) / f'{image_path.stem}.json'
                current_app.logger.debug(
                    f'label file save path is {label_path}'
                )

                with label_path.open(mode='w') as f:
                    json.dump(label, f, ensure_ascii=False, indent=2)
                    return jsonify(label)

        abort(404)

    flash('Please select label file')
    return render_template('file/upload.html')


@bp.route('/<filename>')
@bp.route('/get/<filename>')
def get(filename):
    label_path = Path(current_app.config['UPLOAD_FOLDER']) / filename
    current_app.logger.debug(label_path)
    assert label_path.suffix == '.json', f'File {filename} is the NOT JSON Format'
    data = json.load(label_path.open())
    return jsonify(data)


def _allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['json']