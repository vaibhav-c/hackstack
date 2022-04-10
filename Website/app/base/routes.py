# -*- encoding: utf-8 -*-

from flask import  render_template, redirect, url_for
from app.base import blueprint



@blueprint.route('/')
def route_default():
    return redirect(url_for('home_blueprint.index'))

## Login & Registration

## Errors

@blueprint.errorhandler(403)
def access_forbidden(error):
    return render_template('page-403.html'), 403

@blueprint.errorhandler(404)
def not_found_error(error):
    return render_template('page-404.html'), 404

@blueprint.errorhandler(500)
def internal_error(error):
    return render_template('page-500.html'), 500
