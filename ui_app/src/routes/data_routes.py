"""
Data routes for the Martian Habitat Pathfinder UI.

This module provides routes for:
1. Managing data pipelines
2. Viewing and managing NASA data
3. Visualizing combined data
4. Configuring data processing parameters
"""

import os
import sys
import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from data.enhanced_data_pipeline import EnhancedDataPipeline

# Configure logging
logger = logging.getLogger("data_routes")

# Initialize blueprint
data_bp = Blueprint('data', __name__)

@data_bp.route('/')
def index():
    """Render the data management dashboard."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    
    # Get NASA data directories
    nasa_dir = os.path.join(data_dir, 'nasa')
    nasa_data_types = []
    
    if os.path.exists(nasa_dir):
        nasa_data_types = [d for d in os.listdir(nasa_dir) if os.path.isdir(os.path.join(nasa_dir, d))]
    
    # Get processed data files
    processed_dir = os.path.join(data_dir, 'processed')
    processed_files = []
    
    if os.path.exists(processed_dir):
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('.json')]
    
    # Check if combined data exists
    combined_data_path = os.path.join(processed_dir, 'combined_data.json')
    combined_data_exists = os.path.exists(combined_data_path)
    
    # Check if simulation init data exists
    simulation_init_path = os.path.join(processed_dir, 'simulation_init.json')
    simulation_init_exists = os.path.exists(simulation_init_path)
    
    return render_template(
        'data/index.html',
        title="Data Management",
        nasa_data_types=nasa_data_types,
        processed_files=processed_files,
        combined_data_exists=combined_data_exists,
        simulation_init_exists=simulation_init_exists
    )

@data_bp.route('/fetch_nasa', methods=['POST'])
def fetch_nasa_data():
    """Fetch data from NASA sources."""
    data_types = request.form.getlist('data_types')
    
    if not data_types:
        flash('No data types selected', 'error')
        return redirect(url_for('data.index'))
    
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Fetch NASA data
        results = data_pipeline.fetch_nasa_data(data_types)
        
        flash(f'Successfully fetched NASA data for: {", ".join(data_types)}', 'success')
        return redirect(url_for('data.index'))
    except Exception as e:
        logger.error(f"Error fetching NASA data: {e}")
        flash(f'Error fetching NASA data: {str(e)}', 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/view_nasa/<data_type>')
def view_nasa_data(data_type):
    """View NASA data for a specific type."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    nasa_dir = os.path.join(data_dir, 'nasa', data_type)
    
    if not os.path.exists(nasa_dir):
        flash(f'NASA data directory for {data_type} not found', 'error')
        return redirect(url_for('data.index'))
    
    # Get CSV files
    csv_files = [f for f in os.listdir(nasa_dir) if f.endswith('.csv')]
    
    if not csv_files:
        flash(f'No CSV files found for {data_type}', 'error')
        return redirect(url_for('data.index'))
    
    # Load first CSV file for preview
    csv_path = os.path.join(nasa_dir, csv_files[0])
    df = pd.read_csv(csv_path)
    
    # Generate preview table
    preview_html = df.head(10).to_html(classes='table table-striped table-bordered')
    
    # Generate basic plot
    plt.figure(figsize=(10, 6))
    
    if 'temperature' in df.columns:
        sns.lineplot(data=df, x=df.index, y='temperature')
        plt.title(f'{data_type.upper()} Temperature Data')
        plt.xlabel('Index')
        plt.ylabel('Temperature (°C)')
    elif 'elevation' in df.columns:
        sns.histplot(data=df, x='elevation', bins=30)
        plt.title(f'{data_type.upper()} Elevation Distribution')
        plt.xlabel('Elevation (m)')
        plt.ylabel('Count')
    elif 'thermal_inertia' in df.columns:
        sns.histplot(data=df, x='thermal_inertia', bins=30)
        plt.title(f'{data_type.upper()} Thermal Inertia Distribution')
        plt.xlabel('Thermal Inertia')
        plt.ylabel('Count')
    else:
        # Just plot the first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            sns.lineplot(data=df, x=df.index, y=numeric_cols[0])
            plt.title(f'{data_type.upper()} {numeric_cols[0]} Data')
            plt.xlabel('Index')
            plt.ylabel(numeric_cols[0])
    
    # Save plot to base64 string
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()
    
    # Get metadata if available
    metadata = None
    metadata_files = [f for f in os.listdir(nasa_dir) if f.endswith('_metadata.json')]
    
    if metadata_files:
        metadata_path = os.path.join(nasa_dir, metadata_files[0])
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata for {data_type}: {e}")
    
    return render_template(
        'data/view_nasa.html',
        title=f"NASA {data_type.upper()} Data",
        data_type=data_type,
        csv_files=csv_files,
        preview_html=preview_html,
        plot_data=plot_data,
        metadata=metadata
    )

@data_bp.route('/view_combined')
def view_combined_data():
    """View combined data from all sources."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    combined_data_path = os.path.join(processed_dir, 'combined_data.json')
    
    if not os.path.exists(combined_data_path):
        flash('Combined data file not found', 'error')
        return redirect(url_for('data.index'))
    
    try:
        # Load combined data
        with open(combined_data_path, 'r') as f:
            combined_data = json.load(f)
        
        # Generate environment plots
        env_plots = []
        
        # Temperature plot
        if 'temperature' in combined_data['environment']:
            temp_data = combined_data['environment']['temperature']
            if temp_data:
                plt.figure(figsize=(10, 6))
                
                # Extract values for plotting
                labels = []
                values = []
                
                for key, value in temp_data.items():
                    if isinstance(value, dict) and 'value' in value:
                        labels.append(key)
                        values.append(value['value'])
                    elif isinstance(value, (int, float)):
                        labels.append(key)
                        values.append(value)
                
                if labels and values:
                    plt.bar(labels, values)
                    plt.title('Temperature Data from All Sources')
                    plt.xlabel('Source')
                    plt.ylabel('Temperature (°C)')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    # Save plot to base64 string
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    temp_plot = base64.b64encode(buffer.read()).decode('utf-8')
                    plt.close()
                    
                    env_plots.append({
                        'title': 'Temperature Data',
                        'plot': temp_plot
                    })
        
        # Resource plots
        resource_plots = []
        
        for resource_type in ['water', 'oxygen', 'food', 'power', 'spare_parts']:
            if resource_type in combined_data['resources']:
                resource_data = combined_data['resources'][resource_type]
                if resource_data:
                    plt.figure(figsize=(10, 6))
                    
                    # Extract values for plotting
                    labels = []
                    values = []
                    
                    for key, value in resource_data.items():
                        if isinstance(value, dict) and 'value' in value:
                            labels.append(key)
                            values.append(value['value'])
                        elif isinstance(value, (int, float)):
                            labels.append(key)
                            values.append(value)
                    
                    if labels and values:
                        plt.bar(labels, values)
                        plt.title(f'{resource_type.capitalize()} Data from All Sources')
                        plt.xlabel('Source')
                        plt.ylabel(f'{resource_type.capitalize()} Amount')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Save plot to base64 string
                        buffer = BytesIO()
                        plt.savefig(buffer, format='png')
                        buffer.seek(0)
                        resource_plot = base64.b64encode(buffer.read()).decode('utf-8')
                        plt.close()
                        
                        resource_plots.append({
                            'title': f'{resource_type.capitalize()} Data',
                            'plot': resource_plot
                        })
        
        return render_template(
            'data/view_combined.html',
            title="Combined Data Visualization",
            combined_data=combined_data,
            env_plots=env_plots,
            resource_plots=resource_plots
        )
    except Exception as e:
        logger.error(f"Error viewing combined data: {e}")
        flash(f'Error viewing combined data: {str(e)}', 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/view_simulation_init')
def view_simulation_init():
    """View simulation initialization data."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    simulation_init_path = os.path.join(processed_dir, 'simulation_init.json')
    
    if not os.path.exists(simulation_init_path):
        flash('Simulation initialization file not found', 'error')
        return redirect(url_for('data.index'))
    
    try:
        # Load simulation init data
        with open(simulation_init_path, 'r') as f:
            simulation_init = json.load(f)
        
        return render_template(
            'data/view_simulation_init.html',
            title="Simulation Initialization Data",
            simulation_init=simulation_init
        )
    except Exception as e:
        logger.error(f"Error viewing simulation init data: {e}")
        flash(f'Error viewing simulation init data: {str(e)}', 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/run_pipeline', methods=['POST'])
def run_pipeline():
    """Run the full data pipeline."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Run full pipeline
        results = data_pipeline.run_full_pipeline()
        
        flash('Data pipeline executed successfully', 'success')
        return redirect(url_for('data.index'))
    except Exception as e:
        logger.error(f"Error running data pipeline: {e}")
        flash(f'Error running data pipeline: {str(e)}', 'error')
        return redirect(url_for('data.index'))

@data_bp.route('/download/<file_type>')
def download_data(file_type):
    """Download data files."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    processed_dir = os.path.join(data_dir, 'processed')
    training_dir = os.path.join(data_dir, 'training')
    
    if file_type == 'combined_data':
        file_path = os.path.join(processed_dir, 'combined_data.json')
    elif file_type == 'simulation_init':
        file_path = os.path.join(processed_dir, 'simulation_init.json')
    elif file_type == 'training_data':
        file_path = os.path.join(training_dir, 'combined_ollama_format.txt')
    else:
        flash(f'Invalid file type: {file_type}', 'error')
        return redirect(url_for('data.index'))
    
    if not os.path.exists(file_path):
        flash(f'File not found: {file_type}', 'error')
        return redirect(url_for('data.index'))
    
    return send_file(file_path, as_attachment=True)
