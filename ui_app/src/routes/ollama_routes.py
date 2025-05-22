"""
Ollama integration routes for the Martian Habitat Pathfinder UI.

This module provides routes for:
1. Managing Ollama models
2. Training models with PDF-extracted data
3. Integrating LLMs with RL for decision making
4. Monitoring and evaluating model performance
"""

import os
import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
import requests
from werkzeug.utils import secure_filename

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from models.ollama_integration import OllamaClient, MarsHabitatLLMAgent, MarsHabitatLLMRL

# Configure logging
logger = logging.getLogger("ollama_routes")

# Initialize blueprint
ollama_bp = Blueprint('ollama', __name__)

# Initialize Ollama client
ollama_client = None
llm_agent = None
llm_rl = None

def init_ollama_components():
    """Initialize Ollama components if not already initialized."""
    global ollama_client, llm_agent, llm_rl
    
    if ollama_client is None:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            data_dir = os.path.join(project_root, 'data')
            cache_dir = os.path.join(data_dir, 'llm_cache')
            
            # Ensure cache directory exists
            os.makedirs(cache_dir, exist_ok=True)
            
            # Initialize Ollama client
            ollama_client = OllamaClient(
                base_url="http://localhost:11434",
                default_model="llama2",
                cache_dir=cache_dir
            )
            
            # Initialize LLM agent
            llm_agent = MarsHabitatLLMAgent(
                data_dir=data_dir,
                model_name="llama2",
                cache_dir=cache_dir
            )
            
            # Initialize LLM-RL integration
            llm_rl = MarsHabitatLLMRL(
                data_dir=data_dir,
                model_name="llama2",
                cache_dir=cache_dir
            )
            
            logger.info("Ollama components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Ollama components: {e}")
            return False
    
    return True

@ollama_bp.route('/')
def index():
    """Render the Ollama management dashboard."""
    init_ollama_components()
    
    # Get available models
    models = []
    try:
        models = ollama_client.list_models()
    except Exception as e:
        flash(f"Failed to get models: {e}", "error")
    
    # Get training data files
    training_files = []
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        training_dir = os.path.join(project_root, 'data', 'training')
        
        if os.path.exists(training_dir):
            for filename in os.listdir(training_dir):
                if filename.endswith('_ollama_format.txt') or filename.endswith('_training_data.json'):
                    training_files.append(filename)
    except Exception as e:
        flash(f"Failed to get training files: {e}", "error")
    
    return render_template(
        'ollama/index.html',
        title="Ollama Management",
        models=models,
        training_files=training_files
    )

@ollama_bp.route('/status')
def status():
    """Check Ollama server status."""
    try:
        init_ollama_components()
        models = ollama_client.list_models()
        return jsonify({
            "status": "online",
            "models": models
        })
    except Exception as e:
        logger.error(f"Ollama server status check failed: {e}")
        return jsonify({
            "status": "offline",
            "error": str(e)
        }), 500

@ollama_bp.route('/models')
def list_models():
    """List available Ollama models."""
    try:
        init_ollama_components()
        models = ollama_client.list_models()
        return jsonify({"models": models})
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/pull', methods=['POST'])
def pull_model():
    """Pull a model from Ollama library."""
    model_name = request.form.get('model_name')
    
    if not model_name:
        flash("Model name is required", "error")
        return redirect(url_for('ollama.index'))
    
    try:
        init_ollama_components()
        result = ollama_client.pull_model(model_name)
        flash(f"Model {model_name} pulled successfully", "success")
        return redirect(url_for('ollama.index'))
    except Exception as e:
        logger.error(f"Failed to pull model {model_name}: {e}")
        flash(f"Failed to pull model: {e}", "error")
        return redirect(url_for('ollama.index'))

@ollama_bp.route('/create', methods=['POST'])
def create_model():
    """Create a custom model with Modelfile."""
    model_name = request.form.get('model_name')
    base_model = request.form.get('base_model')
    system_prompt = request.form.get('system_prompt')
    
    if not model_name or not base_model:
        flash("Model name and base model are required", "error")
        return redirect(url_for('ollama.index'))
    
    try:
        # Create Modelfile
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        modelfile_path = os.path.join(project_root, 'data', 'Modelfile')
        
        with open(modelfile_path, 'w') as f:
            f.write(f"FROM {base_model}\n")
            
            if system_prompt:
                f.write(f"SYSTEM {system_prompt}\n")
            else:
                f.write("SYSTEM You are an AI assistant specialized in Mars habitat resource management.\n")
            
            f.write("TEMPLATE <s>[INST] {{ .System }} {{ .Prompt }} [/INST] {{ .Response }}</s>\n")
            f.write("PARAMETER temperature 0.7\n")
            f.write("PARAMETER top_p 0.9\n")
        
        # Create model using Ollama CLI
        import subprocess
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", modelfile_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            flash(f"Model {model_name} created successfully", "success")
        else:
            flash(f"Failed to create model: {result.stderr}", "error")
        
        return redirect(url_for('ollama.index'))
    except Exception as e:
        logger.error(f"Failed to create model {model_name}: {e}")
        flash(f"Failed to create model: {e}", "error")
        return redirect(url_for('ollama.index'))

@ollama_bp.route('/finetune', methods=['POST'])
def finetune_model():
    """Fine-tune a model with training data."""
    model_name = request.form.get('model_name')
    training_file = request.form.get('training_file')
    
    if not model_name or not training_file:
        flash("Model name and training file are required", "error")
        return redirect(url_for('ollama.index'))
    
    try:
        # Get training file path
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        training_dir = os.path.join(project_root, 'data', 'training')
        training_file_path = os.path.join(training_dir, training_file)
        
        if not os.path.exists(training_file_path):
            flash(f"Training file not found: {training_file}", "error")
            return redirect(url_for('ollama.index'))
        
        # Fine-tune model using Ollama CLI
        import subprocess
        result = subprocess.run(
            ["ollama", "run", model_name, "-f", training_file_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            flash(f"Model {model_name} fine-tuned successfully", "success")
        else:
            flash(f"Failed to fine-tune model: {result.stderr}", "error")
        
        return redirect(url_for('ollama.index'))
    except Exception as e:
        logger.error(f"Failed to fine-tune model {model_name}: {e}")
        flash(f"Failed to fine-tune model: {e}", "error")
        return redirect(url_for('ollama.index'))

@ollama_bp.route('/generate', methods=['POST'])
def generate_text():
    """Generate text using an Ollama model."""
    model_name = request.form.get('model_name')
    prompt = request.form.get('prompt')
    temperature = float(request.form.get('temperature', 0.7))
    max_tokens = int(request.form.get('max_tokens', 2048))
    
    if not model_name or not prompt:
        return jsonify({"error": "Model name and prompt are required"}), 400
    
    try:
        init_ollama_components()
        result = ollama_client.generate(
            prompt=prompt,
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return jsonify({
            "response": result.get("response", ""),
            "model": model_name,
            "prompt": prompt
        })
    except Exception as e:
        logger.error(f"Failed to generate text: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/generate_json', methods=['POST'])
def generate_json():
    """Generate structured JSON using an Ollama model."""
    model_name = request.form.get('model_name')
    prompt = request.form.get('prompt')
    schema = request.form.get('schema')
    temperature = float(request.form.get('temperature', 0.2))
    
    if not model_name or not prompt or not schema:
        return jsonify({"error": "Model name, prompt, and schema are required"}), 400
    
    try:
        # Parse schema
        schema_dict = json.loads(schema)
        
        init_ollama_components()
        result = ollama_client.generate_json(
            prompt=prompt,
            model=model_name,
            schema=schema_dict,
            temperature=temperature
        )
        
        return jsonify({
            "parsed_json": result.get("parsed_json", {}),
            "raw_response": result.get("response", ""),
            "model": model_name,
            "prompt": prompt
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON schema"}), 400
    except Exception as e:
        logger.error(f"Failed to generate JSON: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/select_action', methods=['POST'])
def select_action():
    """Select action for Mars habitat management using LLM agent."""
    model_name = request.form.get('model_name')
    state_json = request.form.get('state')
    mode = request.form.get('mode', 'llm')  # llm, rl, or hybrid
    
    if not model_name or not state_json:
        return jsonify({"error": "Model name and state are required"}), 400
    
    try:
        # Parse state
        state = json.loads(state_json)
        
        init_ollama_components()
        
        # Update model name if different
        if llm_rl.llm_agent.model_name != model_name:
            llm_rl.llm_agent.model_name = model_name
        
        # Select action
        action = llm_rl.select_action(state, mode=mode)
        
        # Get explanation
        explanation = llm_rl.explain_action(state, action)
        
        return jsonify({
            "action": action,
            "explanation": explanation,
            "model": model_name,
            "mode": mode
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid state JSON"}), 400
    except Exception as e:
        logger.error(f"Failed to select action: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/generate_scenario', methods=['POST'])
def generate_scenario():
    """Generate a scenario for Mars habitat simulation."""
    model_name = request.form.get('model_name')
    difficulty = request.form.get('difficulty', 'normal')
    scenario_type = request.form.get('scenario_type')
    
    if not model_name:
        return jsonify({"error": "Model name is required"}), 400
    
    try:
        init_ollama_components()
        
        # Update model name if different
        if llm_agent.model_name != model_name:
            llm_agent.model_name = model_name
        
        # Generate scenario
        scenario = llm_agent.generate_scenario(
            difficulty=difficulty,
            scenario_type=scenario_type
        )
        
        return jsonify({
            "scenario": scenario,
            "model": model_name,
            "difficulty": difficulty,
            "scenario_type": scenario_type
        })
    except Exception as e:
        logger.error(f"Failed to generate scenario: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/answer_query', methods=['POST'])
def answer_query():
    """Answer a natural language query about Mars habitat management."""
    model_name = request.form.get('model_name')
    query = request.form.get('query')
    state_json = request.form.get('state')
    
    if not model_name or not query:
        return jsonify({"error": "Model name and query are required"}), 400
    
    try:
        # Parse state if provided
        state = None
        if state_json:
            state = json.loads(state_json)
        
        init_ollama_components()
        
        # Update model name if different
        if llm_agent.model_name != model_name:
            llm_agent.model_name = model_name
        
        # Answer query
        answer = llm_agent.answer_query(query, state)
        
        return jsonify({
            "answer": answer,
            "model": model_name,
            "query": query
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid state JSON"}), 400
    except Exception as e:
        logger.error(f"Failed to answer query: {e}")
        return jsonify({"error": str(e)}), 500

@ollama_bp.route('/create_training_data', methods=['POST'])
def create_training_data():
    """Create training data for Ollama fine-tuning."""
    num_examples = int(request.form.get('num_examples', 100))
    
    try:
        init_ollama_components()
        
        # Create training data
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        output_path = os.path.join(project_root, 'data', 'training', f"generated_training_data_{num_examples}.txt")
        
        training_data_path = llm_rl.create_training_data(
            num_examples=num_examples,
            output_path=output_path
        )
        
        flash(f"Training data created successfully: {os.path.basename(training_data_path)}", "success")
        return redirect(url_for('ollama.index'))
    except Exception as e:
        logger.error(f"Failed to create training data: {e}")
        flash(f"Failed to create training data: {e}", "error")
        return redirect(url_for('ollama.index'))

@ollama_bp.route('/test', methods=['GET'])
def test_page():
    """Render the Ollama test page."""
    init_ollama_components()
    
    # Get available models
    models = []
    try:
        models = ollama_client.list_models()
    except Exception as e:
        flash(f"Failed to get models: {e}", "error")
    
    return render_template(
        'ollama/test.html',
        title="Test Ollama Integration",
        models=models
    )
