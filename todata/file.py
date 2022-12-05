import os
from flask import (
    Blueprint, current_app, flash, g,
    redirect, render_template, request,
    send_from_directory, url_for
)
from werkzeug.utils import secure_filename

bp = Blueprint('file', __name__, url_prefix='/file')


def allowed_file(filename: str) -> bool:
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']


@bp.route('/upload', methods=('GET', 'POST'))
def upload():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if the user does not select a file, the browser submits an
        # empty file without a filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(
                os.path.join(
                    current_app.config['UPLOAD_FOLDER'], filename
                )
            )
            # file.stream.read() # type(file.stream.read()) is <class 'bytes'>
            current_app.logger.debug(
                f'file save path is {os.path.join(current_app.config["UPLOAD_FOLDER"], filename)}'
            )
            return redirect(url_for('file.download', filename=filename))
    return render_template('file/upload.html')


@bp.route('/<filename>')
@bp.route('/download/<filename>')
def download(filename):
    return send_from_directory(current_app.config['UPLOAD_FOLDER'], filename)
