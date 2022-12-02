import os

from flask import Flask


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE = os.path.join(app.instance_path, 'todata.sqlite'),
        UPLOAD_FOLDER = r'D:\PycharmProjects\todata\upload',
        ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'},
        MAX_CONTENT_LENGTH = 16 * 1000 * 1000, # less than 16MB
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db
    db.init_app(app)

    from . import file
    app.register_blueprint(file.bp)


    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello World!'

    return app