"""
PDF processing routes for the Martian Habitat Pathfinder UI.

This module provides routes for:
1. Uploading PDF documents
2. Processing PDFs to extract scientific data
3. Converting PDFs to training data for LLMs
4. Managing PDF documents and their extracted data
"""

import os
import sys
import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app, send_file
from werkzeug.utils import secure_filename

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from utils.pdf_processor import PDFProcessor
from data.enhanced_data_pipeline import EnhancedDataPipeline

# Configure logging
logger = logging.getLogger("pdf_routes")

# Initialize blueprint
pdf_bp = Blueprint('pdf', __name__)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

@pdf_bp.route('/')
def index():
    """Render the PDF management dashboard."""
    # Get list of uploaded PDFs
    pdf_dir = current_app.config['UPLOAD_FOLDER']
    pdfs = []
    
    if os.path.exists(pdf_dir):
        pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]
    
    # Get list of processed data
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    processed_files = []
    
    if os.path.exists(processed_dir):
        processed_files = [f for f in os.listdir(processed_dir) if f.endswith('_scientific_data.json') or f.endswith('_simulation_params.json')]
    
    # Get list of training data
    training_dir = os.path.join(project_root, 'data', 'training')
    training_files = []
    
    if os.path.exists(training_dir):
        training_files = [f for f in os.listdir(training_dir) if f.endswith('_training_data.json') or f.endswith('_ollama_format.txt')]
    
    return render_template(
        'pdf/index.html',
        title="PDF Management",
        pdfs=pdfs,
        processed_files=processed_files,
        training_files=training_files
    )

@pdf_bp.route('/upload', methods=['POST'])
def upload_pdf():
    """Upload a PDF file."""
    # Check if the post request has the file part
    if 'pdf_file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('pdf.index'))
    
    file = request.files['pdf_file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('pdf.index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(file_path)
        
        flash(f'File {filename} uploaded successfully', 'success')
        
        # Check if auto-process is requested
        auto_process = request.form.get('auto_process') == 'on'
        
        if auto_process:
            return redirect(url_for('pdf.process_pdf', filename=filename))
        else:
            return redirect(url_for('pdf.index'))
    else:
        flash('File type not allowed. Please upload a PDF file.', 'error')
        return redirect(url_for('pdf.index'))

@pdf_bp.route('/process/<filename>', methods=['GET', 'POST'])
def process_pdf(filename):
    """Process a PDF file to extract data."""
    file_path = os.path.normpath(os.path.join(current_app.config['UPLOAD_FOLDER'], filename))
    
    if not file_path.startswith(os.path.abspath(current_app.config['UPLOAD_FOLDER'])):
        flash('Invalid file path', 'error')
        return redirect(url_for('pdf.index'))
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('pdf.index'))
    
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Process PDF
        results = data_pipeline.process_pdf_file(file_path)
        
        flash(f'File {filename} processed successfully', 'success')
        
        # Redirect to view results
        return redirect(url_for('pdf.view_results', filename=filename))
    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {e}")
        flash(f'Error processing file: {str(e)}', 'error')
        return redirect(url_for('pdf.index'))

@pdf_bp.route('/view/<filename>')
def view_pdf(filename):
    """View a PDF file."""
    file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('pdf.index'))
    
    return render_template(
        'pdf/view.html',
        title=f"View PDF: {filename}",
        filename=filename
    )

@pdf_bp.route('/download/<filename>')
def download_pdf(filename):
    """Download a PDF file."""
    sanitized_filename = secure_filename(filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.normpath(os.path.join(upload_folder, sanitized_filename))
    
    # Ensure the file path is within the UPLOAD_FOLDER
    if not file_path.startswith(os.path.abspath(upload_folder)):
        flash('Invalid file path', 'error')
        return redirect(url_for('pdf.index'))
    
    if not os.path.exists(file_path):
        flash(f'File {sanitized_filename} not found', 'error')
        return redirect(url_for('pdf.index'))
    
    return send_file(file_path, as_attachment=True)

@pdf_bp.route('/delete/<filename>', methods=['POST'])
def delete_pdf(filename):
    """Delete a PDF file."""
    sanitized_filename = secure_filename(filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.normpath(os.path.join(upload_folder, sanitized_filename))
    
    # Ensure the file path is within the UPLOAD_FOLDER
    if not file_path.startswith(os.path.abspath(upload_folder)):
        flash('Invalid file path', 'error')
        return redirect(url_for('pdf.index'))
    
    if not os.path.exists(file_path):
        flash(f'File {sanitized_filename} not found', 'error')
        return redirect(url_for('pdf.index'))
    
    try:
        os.remove(file_path)
        flash(f'File {sanitized_filename} deleted successfully', 'success')
    except Exception as e:
        logger.error(f"Error deleting PDF {sanitized_filename}: {e}")
        flash(f'Error deleting file: {str(e)}', 'error')
    
    return redirect(url_for('pdf.index'))

@pdf_bp.route('/results/<filename>')
def view_results(filename):
    """View processing results for a PDF file."""
    # Sanitize and validate filename
    from werkzeug.utils import secure_filename
    filename = secure_filename(filename)
    base_name = os.path.splitext(filename)[0]
    
    # Get paths to result files
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    training_dir = os.path.join(project_root, 'data', 'training')
    
    scientific_data_path = os.path.normpath(os.path.join(processed_dir, f"{base_name}_scientific_data.json"))
    simulation_params_path = os.path.normpath(os.path.join(processed_dir, f"{base_name}_simulation_params.json"))
    training_data_path = os.path.normpath(os.path.join(training_dir, f"{base_name}_training_data.json"))
    ollama_format_path = os.path.normpath(os.path.join(training_dir, f"{base_name}_ollama_format.txt"))
    
    # Ensure paths are within their respective directories
    if not scientific_data_path.startswith(processed_dir) or \
       not simulation_params_path.startswith(processed_dir) or \
       not training_data_path.startswith(training_dir) or \
       not ollama_format_path.startswith(training_dir):
        logger.error(f"Invalid path traversal attempt detected for filename: {filename}")
        flash("Invalid file path.", "error")
        return redirect(url_for('pdf.index'))
    
    # Load data if files exist
    scientific_data = None
    simulation_params = None
    training_data_info = None
    
    if os.path.exists(scientific_data_path):
        try:
            with open(scientific_data_path, 'r') as f:
                scientific_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading scientific data for {filename}: {e}")
    
    if os.path.exists(simulation_params_path):
        try:
            with open(simulation_params_path, 'r') as f:
                simulation_params = json.load(f)
        except Exception as e:
            logger.error(f"Error loading simulation parameters for {filename}: {e}")
    
    if os.path.exists(training_data_path):
        try:
            with open(training_data_path, 'r') as f:
                training_data = json.load(f)
                training_data_info = {
                    'count': len(training_data),
                    'file_size': os.path.getsize(training_data_path),
                    'path': training_data_path
                }
        except Exception as e:
            logger.error(f"Error loading training data for {filename}: {e}")
    
    return render_template(
        'pdf/results.html',
        title=f"Processing Results: {filename}",
        filename=filename,
        scientific_data=scientific_data,
        simulation_params=simulation_params,
        training_data_info=training_data_info,
        ollama_format_exists=os.path.exists(ollama_format_path)
    )

@pdf_bp.route('/download_results/<filename>/<result_type>')
def download_results(filename, result_type):
    """Download processing results for a PDF file."""
    from werkzeug.utils import secure_filename
    
    # Sanitize the filename
    filename = secure_filename(filename)
    base_name = os.path.splitext(filename)[0]
    
    # Get paths to result files
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    processed_dir = os.path.join(project_root, 'data', 'processed')
    training_dir = os.path.join(project_root, 'data', 'training')
    
    if result_type == 'scientific_data':
        file_path = os.path.join(processed_dir, f"{base_name}_scientific_data.json")
    elif result_type == 'simulation_params':
        file_path = os.path.join(processed_dir, f"{base_name}_simulation_params.json")
    elif result_type == 'training_data':
        file_path = os.path.join(training_dir, f"{base_name}_training_data.json")
    elif result_type == 'ollama_format':
        file_path = os.path.join(training_dir, f"{base_name}_ollama_format.txt")
    else:
        flash(f'Invalid result type: {result_type}', 'error')
        return redirect(url_for('pdf.view_results', filename=filename))
    
    # Normalize and validate the file path
    file_path = os.path.normpath(file_path)
    if not (file_path.startswith(processed_dir) or file_path.startswith(training_dir)):
        flash('Invalid file path', 'error')
        return redirect(url_for('pdf.view_results', filename=filename))
    
    if not os.path.exists(file_path):
        flash(f'Result file not found', 'error')
        return redirect(url_for('pdf.view_results', filename=filename))
    
    return send_file(file_path, as_attachment=True)

@pdf_bp.route('/process_all', methods=['POST'])
def process_all_pdfs():
    """Process all uploaded PDF files."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Process all PDFs
        results = data_pipeline.process_pdf_directory(current_app.config['UPLOAD_FOLDER'])
        
        # Combine data sources
        combined_data = data_pipeline.combine_data_sources(pdf_results=results)
        
        # Prepare training data for Ollama
        training_data_path = data_pipeline.prepare_training_data_for_ollama()
        
        # Prepare simulation data
        simulation_data = data_pipeline.prepare_simulation_data()
        
        flash(f'Processed {len(results["processed_files"])} PDF files successfully', 'success')
        return redirect(url_for('pdf.index'))
    except Exception as e:
        logger.error(f"Error processing PDFs: {e}")
        flash(f'Error processing files: {str(e)}', 'error')
        return redirect(url_for('pdf.index'))

@pdf_bp.route('/combine_data', methods=['POST'])
def combine_data():
    """Combine data from all processed PDFs."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Get processed PDF results
        processed_dir = os.path.join(project_root, 'data', 'processed')
        results_path = os.path.join(processed_dir, "pdf_processing_results.json")
        
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
        else:
            # If no results file exists, process all PDFs
            results = data_pipeline.process_pdf_directory(current_app.config['UPLOAD_FOLDER'])
        
        # Combine data sources
        combined_data = data_pipeline.combine_data_sources(pdf_results=results)
        
        # Prepare simulation data
        simulation_data = data_pipeline.prepare_simulation_data()
        
        flash('Data combined successfully', 'success')
        return redirect(url_for('pdf.index'))
    except Exception as e:
        logger.error(f"Error combining data: {e}")
        flash(f'Error combining data: {str(e)}', 'error')
        return redirect(url_for('pdf.index'))

@pdf_bp.route('/prepare_training', methods=['POST'])
def prepare_training_data():
    """Prepare training data for Ollama from all processed PDFs."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Prepare training data for Ollama
        training_data_path = data_pipeline.prepare_training_data_for_ollama()
        
        flash(f'Training data prepared successfully: {os.path.basename(training_data_path)}', 'success')
        return redirect(url_for('pdf.index'))
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        flash(f'Error preparing training data: {str(e)}', 'error')
        return redirect(url_for('pdf.index'))

@pdf_bp.route('/prepare_simulation', methods=['POST'])
def prepare_simulation_data():
    """Prepare simulation data from all processed PDFs."""
    # Initialize data pipeline
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_root, 'data')
    data_pipeline = EnhancedDataPipeline(data_dir)
    
    try:
        # Prepare simulation data
        simulation_data = data_pipeline.prepare_simulation_data()
        
        flash('Simulation data prepared successfully', 'success')
        return redirect(url_for('pdf.index'))
    except Exception as e:
        logger.error(f"Error preparing simulation data: {e}")
        flash(f'Error preparing simulation data: {str(e)}', 'error')
        return redirect(url_for('pdf.index'))
