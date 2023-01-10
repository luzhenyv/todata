"""Create and Initialize the ToData Flask APP"""
import os

from flask import Flask, jsonify, request, abort, current_app, make_response

from todata.blueprints.main import main_bp
from todata.settings import config


def create_app(config_name=None):
    """ The Application Factory

    Attributes:
        config_name: a string indicating the config name
            including 'development', 'testing', 'production'.
    Returns:
        app: a ToData flask application instance
    """
    if config_name is None:
        config_name = os.getenv('FLASK_CONFIG', 'development')

    app = Flask('todata')

    app.config.from_object(config[config_name])

    return app


def register_blueprints(app):
    """Register Blueprints"""
    app.register_blueprint(main_bp)

