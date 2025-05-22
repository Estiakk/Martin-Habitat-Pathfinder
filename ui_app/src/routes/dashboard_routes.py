"""
Dashboard routes for the Martian Habitat Pathfinder UI.

This module provides routes for:
1. Main dashboard overview
2. System status monitoring
3. Quick access to all major features
4. Integrated workflow visualization
"""

import os
import sys
import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
import psutil

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Configure logging
logger = logging.getLogger("dashboard_routes")

# Initialize blueprint
dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/')
def index():
    """Render the main dashboard."""
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    
    # Get system status
    system_status = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }
    
    # Get PDF status
    pdf_dir = current_app.config['UPLOAD_FOLDER']
    pdf_count = 0
    if os.path.exists(pdf_dir):
        pdf_count = len([f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])
    
    # Get data status
    processed_dir = os.path.join(data_dir, 'processed')
    processed_count = 0
    if os.path.exists(processed_dir):
        processed_count = len([f for f in os.listdir(processed_dir) if f.endswith('.json')])
    
    # Get training data status
    training_dir = os.path.join(data_dir, 'training')
    training_count = 0
    if os.path.exists(training_dir):
        training_count = len([f for f in os.listdir(training_dir) if f.endswith('.json') or f.endswith('.txt')])
    
    # Get Ollama status
    ollama_status = "Unknown"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            ollama_status = "Online"
            ollama_models = len(response.json().get('models', []))
        else:
            ollama_status = "Error"
            ollama_models = 0
    except Exception as e:
        logger.error(f"Error checking Ollama status: {e}")
        ollama_status = "Offline"
        ollama_models = 0
    
    # Get simulation status
    simulation_init_path = os.path.join(processed_dir, 'simulation_init.json')
    simulation_status = "Not Configured"
    if os.path.exists(simulation_init_path):
        simulation_status = "Ready"
    
    # Get recent activities
    recent_activities = []
    
    # Check for recent PDF uploads
    if os.path.exists(pdf_dir):
        for filename in sorted(os.listdir(pdf_dir), key=lambda x: os.path.getmtime(os.path.join(pdf_dir, x)), reverse=True)[:5]:
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(pdf_dir, filename)
                recent_activities.append({
                    'type': 'PDF Upload',
                    'name': filename,
                    'time': os.path.getmtime(file_path),
                    'url': url_for('pdf.view_pdf', filename=filename)
                })
    
    # Check for recent processed files
    if os.path.exists(processed_dir):
        for filename in sorted(os.listdir(processed_dir), key=lambda x: os.path.getmtime(os.path.join(processed_dir, x)), reverse=True)[:5]:
            if filename.endswith('_scientific_data.json'):
                file_path = os.path.join(processed_dir, filename)
                pdf_name = filename.replace('_scientific_data.json', '.pdf')
                recent_activities.append({
                    'type': 'Data Processing',
                    'name': pdf_name,
                    'time': os.path.getmtime(file_path),
                    'url': url_for('pdf.view_results', filename=pdf_name)
                })
    
    # Sort activities by time
    recent_activities = sorted(recent_activities, key=lambda x: x['time'], reverse=True)[:5]
    
    # Format times
    import datetime
    for activity in recent_activities:
        activity['time_str'] = datetime.datetime.fromtimestamp(activity['time']).strftime('%Y-%m-%d %H:%M:%S')
    
    return render_template(
        'dashboard/index.html',
        title="Martian Habitat Pathfinder Dashboard",
        system_status=system_status,
        pdf_count=pdf_count,
        processed_count=processed_count,
        training_count=training_count,
        ollama_status=ollama_status,
        ollama_models=ollama_models,
        simulation_status=simulation_status,
        recent_activities=recent_activities
    )

@dashboard_bp.route('/status')
def status():
    """Get system status as JSON."""
    # Get system status
    system_status = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'disk_percent': psutil.disk_usage('/').percent
    }
    
    # Get Ollama status
    ollama_status = "Unknown"
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            ollama_status = "Online"
            ollama_models = len(response.json().get('models', []))
        else:
            ollama_status = "Error"
            ollama_models = 0
    except Exception as e:
        logger.error(f"Error checking Ollama status: {e}")
        ollama_status = "Offline"
        ollama_models = 0
    
    return jsonify({
        'system': system_status,
        'ollama': {
            'status': ollama_status,
            'models': ollama_models
        },
        'timestamp': int(time.time())
    })

@dashboard_bp.route('/workflow')
def workflow():
    """Render the workflow visualization page."""
    return render_template(
        'dashboard/workflow.html',
        title="Workflow Visualization"
    )

@dashboard_bp.route('/quick_start')
def quick_start():
    """Render the quick start guide."""
    return render_template(
        'dashboard/quick_start.html',
        title="Quick Start Guide"
    )
