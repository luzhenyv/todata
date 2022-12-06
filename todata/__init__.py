import os

from flask import Flask, jsonify, request, abort, current_app, make_response

tasks = [
    {
        'id': 1,
        'title': u'Buy groceries',
        'description': u'Milk, Cheese, Pizza, Fruit, Tylenol',
        'done': False
    },
    {
        'id': 2,
        'title': u'Learn Python',
        'description': u'Need to find a good Python tutorial on the web',
        'done': False
    }
]


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE = os.path.join(app.instance_path, 'todata.sqlite'),
        UPLOAD_FOLDER = r'../upload',
        ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'},
        MAX_CONTENT_LENGTH = 16 * 1000 * 1000, # less than 16MB
        MODEL_WEIGHT_PATH = r'../weight/best.onnx'
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

    from . import label
    app.register_blueprint(label.bp)


    # a simple page that says hello
    @app.route('/hello')
    def hello():
        return 'Hello World!'

    @app.route('/todata/api/v1.0/tasks', methods=['GET'])
    def get_tasks():
        return jsonify({'tasks': tasks})

    @app.route('/todata/api/v1.0/tasks/<int:task_id>', methods=['GET'])
    def get_task(task_id):
        task = list(filter(lambda t: t['id'] == task_id, tasks))
        current_app.logger.info(task)
        if len(task) == 0:
            abort(404)
        return jsonify({'task': task[0]})

    @app.route('/todata/api/v1.0/tasks', methods=['POST'])
    def create_task():
        if not request.json or not 'title' in request.json:
            abort(400)
        task = {
            'id': tasks[-1]['id'] + 1,
            'title': request.json['title'],
            'description': request.json.get('description', ""),
            'done': False
        }
        tasks.append(task)
        return jsonify({'task': task}), 201


    return app