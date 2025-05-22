"""
Simulation routes for the Martian Habitat Pathfinder UI.

This module provides routes for:
1. Configuring simulation parameters
2. Running simulations with different scenarios
3. Visualizing simulation results
4. Integrating RL agents with simulations
"""

import os
import sys
import json
import logging
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app
from werkzeug.utils import secure_filename

# Import project modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from simulations.simulation_environment import MarsHabitatSimulation
from simulations.simulation_llm_bridge import SimulationLLMBridge
from models.baseline_rl import DQNAgent, PPOAgent
from models.advanced_rl import HierarchicalRLAgent, MultiAgentRLSystem

# Configure logging
logger = logging.getLogger("simulation_routes")

# Initialize blueprint
simulation_bp = Blueprint('simulation', __name__)

# Initialize simulation components
simulation = None
llm_bridge = None
rl_agents = {}

def init_simulation_components():
    """Initialize simulation components if not already initialized."""
    global simulation, llm_bridge, rl_agents
    
    if simulation is None:
        try:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            data_dir = os.path.join(project_root, 'data')
            config_path = os.path.join(data_dir, 'processed', 'simulation_init.json')
            
            # Load simulation config if exists, otherwise use defaults
            config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
            
            # Initialize simulation
            simulation = MarsHabitatSimulation(config=config)
            
            # Initialize LLM bridge
            llm_bridge = SimulationLLMBridge(
                simulation=simulation,
                data_dir=data_dir
            )
            
            # Initialize RL agents
            rl_agents = {
                'dqn': DQNAgent(simulation.get_observation_space(), simulation.get_action_space()),
                'ppo': PPOAgent(simulation.get_observation_space(), simulation.get_action_space()),
                'hierarchical': HierarchicalRLAgent(simulation.get_observation_space(), simulation.get_action_space()),
                'multi_agent': MultiAgentRLSystem(simulation.get_observation_space(), simulation.get_action_space())
            }
            
            logger.info("Simulation components initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize simulation components: {e}")
            return False
    
    return True

@simulation_bp.route('/')
def index():
    """Render the simulation dashboard."""
    init_simulation_components()
    
    # Get available scenarios
    scenarios = simulation.get_available_scenarios()
    
    # Get available RL agents
    agents = list(rl_agents.keys())
    
    # Get current simulation config
    config = simulation.get_config()
    
    return render_template(
        'simulation/index.html',
        title="Simulation Dashboard",
        scenarios=scenarios,
        agents=agents,
        config=config
    )

@simulation_bp.route('/config', methods=['GET', 'POST'])
def config():
    """Configure simulation parameters."""
    init_simulation_components()
    
    if request.method == 'POST':
        try:
            # Get form data
            config = {
                'environment': {
                    'temperature_range': [
                        float(request.form.get('min_temperature', -120)),
                        float(request.form.get('max_temperature', 20))
                    ],
                    'pressure_range': [
                        float(request.form.get('min_pressure', 600)),
                        float(request.form.get('max_pressure', 700))
                    ],
                    'dust_opacity_range': [
                        float(request.form.get('min_dust_opacity', 0.1)),
                        float(request.form.get('max_dust_opacity', 0.9))
                    ],
                    'solar_irradiance_range': [
                        float(request.form.get('min_solar_irradiance', 0)),
                        float(request.form.get('max_solar_irradiance', 600))
                    ]
                },
                'habitat': {
                    'initial_power': float(request.form.get('initial_power', 100)),
                    'initial_water': float(request.form.get('initial_water', 500)),
                    'initial_oxygen': float(request.form.get('initial_oxygen', 200)),
                    'initial_food': float(request.form.get('initial_food', 300)),
                    'initial_spare_parts': float(request.form.get('initial_spare_parts', 50))
                },
                'simulation': {
                    'max_steps': int(request.form.get('max_steps', 500)),
                    'difficulty': request.form.get('difficulty', 'normal')
                }
            }
            
            # Update simulation config
            simulation.update_config(config)
            
            # Save config to file
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            config_path = os.path.join(project_root, 'data', 'processed', 'simulation_init.json')
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            flash("Simulation configuration updated successfully", "success")
            return redirect(url_for('simulation.index'))
        except Exception as e:
            logger.error(f"Failed to update simulation config: {e}")
            flash(f"Failed to update configuration: {e}", "error")
            return redirect(url_for('simulation.config'))
    
    # Get current config
    config = simulation.get_config()
    
    return render_template(
        'simulation/config.html',
        title="Configure Simulation",
        config=config
    )

@simulation_bp.route('/run', methods=['POST'])
def run_simulation():
    """Run a simulation with specified parameters."""
    init_simulation_components()
    
    scenario = request.form.get('scenario', 'default')
    agent_type = request.form.get('agent_type', 'human')
    steps = int(request.form.get('steps', 100))
    
    try:
        # Reset simulation with selected scenario
        observation = simulation.reset(scenario=scenario)
        
        results = {
            'steps': [],
            'final_state': None,
            'metrics': None
        }
        
        # Run simulation
        if agent_type == 'human':
            # Manual mode - just initialize and return first observation
            results['initial_observation'] = observation
            results['scenario'] = scenario
            
            return jsonify(results)
        else:
            # Agent mode - run for specified steps
            agent = rl_agents.get(agent_type)
            
            if agent is None:
                return jsonify({"error": f"Agent type {agent_type} not found"}), 400
            
            # Run simulation with agent
            for i in range(steps):
                # Get action from agent
                action = agent.select_action(observation)
                
                # Take step in environment
                next_observation, reward, done, info = simulation.step(action)
                
                # Record step results
                results['steps'].append({
                    'step': i,
                    'observation': observation,
                    'action': action,
                    'reward': reward,
                    'info': info
                })
                
                # Update observation
                observation = next_observation
                
                # Check if done
                if done:
                    break
            
            # Get final state and metrics
            results['final_state'] = simulation.get_state()
            results['metrics'] = simulation.get_metrics()
            
            return jsonify(results)
    except Exception as e:
        logger.error(f"Failed to run simulation: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/step', methods=['POST'])
def simulation_step():
    """Take a single step in the simulation."""
    init_simulation_components()
    
    action_json = request.form.get('action')
    
    if not action_json:
        return jsonify({"error": "Action is required"}), 400
    
    try:
        # Parse action
        action = json.loads(action_json)
        
        # Take step in environment
        observation, reward, done, info = simulation.step(action)
        
        # Get current state
        state = simulation.get_state()
        
        return jsonify({
            'observation': observation,
            'reward': reward,
            'done': done,
            'info': info,
            'state': state
        })
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid action JSON"}), 400
    except Exception as e:
        logger.error(f"Failed to take simulation step: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/reset', methods=['POST'])
def reset_simulation():
    """Reset the simulation with specified scenario."""
    init_simulation_components()
    
    scenario = request.form.get('scenario', 'default')
    
    try:
        # Reset simulation
        observation = simulation.reset(scenario=scenario)
        
        # Get current state
        state = simulation.get_state()
        
        return jsonify({
            'observation': observation,
            'state': state,
            'scenario': scenario
        })
    except Exception as e:
        logger.error(f"Failed to reset simulation: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/state')
def get_simulation_state():
    """Get the current simulation state."""
    init_simulation_components()
    
    try:
        # Get current state
        state = simulation.get_state()
        
        return jsonify(state)
    except Exception as e:
        logger.error(f"Failed to get simulation state: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/metrics')
def get_simulation_metrics():
    """Get the current simulation metrics."""
    init_simulation_components()
    
    try:
        # Get metrics
        metrics = simulation.get_metrics()
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Failed to get simulation metrics: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/scenarios')
def get_scenarios():
    """Get available simulation scenarios."""
    init_simulation_components()
    
    try:
        # Get available scenarios
        scenarios = simulation.get_available_scenarios()
        
        return jsonify({"scenarios": scenarios})
    except Exception as e:
        logger.error(f"Failed to get scenarios: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/llm_action', methods=['POST'])
def get_llm_action():
    """Get action recommendation from LLM."""
    init_simulation_components()
    
    model_name = request.form.get('model_name', 'llama2')
    
    try:
        # Get current state
        state = simulation.get_state()
        
        # Get action from LLM
        action, explanation = llm_bridge.get_llm_action(state, model_name)
        
        return jsonify({
            'action': action,
            'explanation': explanation,
            'state': state
        })
    except Exception as e:
        logger.error(f"Failed to get LLM action: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/train_rl', methods=['POST'])
def train_rl_agent():
    """Train an RL agent on the simulation."""
    init_simulation_components()
    
    agent_type = request.form.get('agent_type')
    episodes = int(request.form.get('episodes', 100))
    
    if not agent_type:
        return jsonify({"error": "Agent type is required"}), 400
    
    if agent_type not in rl_agents:
        return jsonify({"error": f"Agent type {agent_type} not found"}), 400
    
    try:
        # Get agent
        agent = rl_agents[agent_type]
        
        # Train agent
        training_results = agent.train(simulation, episodes=episodes)
        
        # Save agent
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        save_path = os.path.join(project_root, 'models', f"{agent_type}_agent.pkl")
        agent.save(save_path)
        
        return jsonify({
            'agent_type': agent_type,
            'episodes': episodes,
            'training_results': training_results,
            'save_path': save_path
        })
    except Exception as e:
        logger.error(f"Failed to train RL agent: {e}")
        return jsonify({"error": str(e)}), 500

@simulation_bp.route('/visualize')
def visualize():
    """Render the simulation visualization page."""
    init_simulation_components()
    
    return render_template(
        'simulation/visualize.html',
        title="Simulation Visualization"
    )
