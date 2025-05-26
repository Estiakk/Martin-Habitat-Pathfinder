"""
Main entry point for the Martian Habitat Pathfinder UI.

This module initializes the Flask application, registers all blueprints,
and configures the application settings.
"""

import os
import sys
import logging
from flask import Flask, render_template, send_from_directory, redirect, url_for

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Configure app
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'uploads')
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Import routes
from src.routes.dashboard_routes import dashboard_bp
from src.routes.pdf_routes import pdf_bp
from src.routes.data_routes import data_bp
from src.routes.simulation_routes import simulation_bp
from src.routes.ollama_routes import ollama_bp

# Register blueprints
app.register_blueprint(dashboard_bp, url_prefix='/dashboard')
app.register_blueprint(pdf_bp, url_prefix='/pdf')
app.register_blueprint(data_bp, url_prefix='/data')
app.register_blueprint(simulation_bp, url_prefix='/simulation')
app.register_blueprint(ollama_bp, url_prefix='/ollama')

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(app.root_path, 'static'), filename)

# Root route
@app.route('/')
def index():
    return redirect(url_for('dashboard.index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_code=404, error_message='Page not found'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error_code=500, error_message='Internal server error'), 500

# Run the app if executed directly
if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_ENV', 'production') == 'development'
    app.run(host='0.0.0.0', port=5000, debug=debug_mode)
