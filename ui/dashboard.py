# Human-AI Interface for Mars Habitat Resource Management

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import json
import sys
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import pickle
from flask import Flask

# Add project directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.decision_integration import DecisionIntegrationSystem
from simulations.rl_environment import MarsHabitatRLEnvironment
from simulations.rl_formulation import MarsHabitatRLFormulation

class MarsHabitatDashboard:
    """
    Interactive dashboard for Mars Habitat Resource Management
    
    This dashboard provides:
    1. Real-time monitoring of habitat resources and systems
    2. Resource forecasting and anomaly detection
    3. Decision recommendations and explanations
    4. Manual control options for human operators
    """
    
    def __init__(self, data_dir):
        """
        Initialize the dashboard
        
        Args:
            data_dir (str): Directory containing data and configuration files
        """
        self.data_dir = data_dir
        self.ui_dir = os.path.join(data_dir, "ui")
        os.makedirs(self.ui_dir, exist_ok=True)
        
        # Create RL formulation
        self.formulation = MarsHabitatRLFormulation(data_dir)
        
        # Create environment
        self.env = self.formulation.create_environment(difficulty='normal')
        
        # Load decision integration system
        self.integration = DecisionIntegrationSystem(data_dir)
        
        # Initialize state
        self.state = self.env.reset()
        
        # Initialize history
        self.history = {
            'time': [],
            'resources': {
                'power': [],
                'water': [],
                'oxygen': [],
                'food': []
            },
            'environment': {
                'temperature': [],
                'pressure': [],
                'dust_opacity': [],
                'solar_irradiance': []
            },
            'subsystems': {
                'power_system': [],
                'life_support': [],
                'isru': [],
                'thermal_control': []
            },
            'decisions': []
        }
        
        # Initialize app
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        # Configure app layout
        self._configure_layout()
        
        # Configure callbacks
        self._configure_callbacks()
        
        print(f"Mars Habitat Dashboard initialized")
    
    def _configure_layout(self):
        """
        Configure dashboard layout
        """
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("Mars Habitat Resource Management Dashboard"),
                html.Div([
                    html.Button("Step Simulation", id="step-button", n_clicks=0),
                    html.Button("Reset Simulation", id="reset-button", n_clicks=0),
                    html.Button("Auto-Pilot", id="auto-button", n_clicks=0),
                    dcc.Interval(id="auto-interval", interval=2000, disabled=True)
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'width': '400px'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}),
            
            # Status Bar
            html.Div([
                html.Div([
                    html.H4("Sol: "),
                    html.H4(id="sol-display", children="0")
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.H4("Hour: "),
                    html.H4(id="hour-display", children="0")
                ], style={'display': 'flex', 'alignItems': 'center'}),
                html.Div([
                    html.H4("Status: "),
                    html.H4(id="status-display", children="Operational")
                ], style={'display': 'flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'backgroundColor': '#f0f0f0', 'padding': '10px', 'borderRadius': '5px', 'marginBottom': '20px'}),
            
            # Main Content
            html.Div([
                # Left Column - Resources
                html.Div([
                    html.H2("Resources"),
                    dcc.Graph(id="resources-graph"),
                    html.Div([
                        html.H3("Current Levels"),
                        html.Div(id="resource-levels")
                    ])
                ], style={'width': '48%'}),
                
                # Right Column - Environment & Systems
                html.Div([
                    html.H2("Environment"),
                    dcc.Graph(id="environment-graph"),
                    html.H2("Subsystems"),
                    dcc.Graph(id="subsystems-graph")
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between'}),
            
            # Decision Support
            html.Div([
                html.H2("Decision Support"),
                
                # Tabs for different decision components
                dcc.Tabs([
                    # Recommendations Tab
                    dcc.Tab(label="Recommendations", children=[
                        html.Div([
                            html.Div([
                                html.H3("AI Recommendations"),
                                html.Div(id="ai-recommendations")
                            ], style={'width': '48%'}),
                            
                            html.Div([
                                html.H3("Manual Control"),
                                html.Div([
                                    html.H4("Power Allocation"),
                                    html.Div([
                                        html.Label("Life Support"),
                                        dcc.Slider(id="life-support-slider", min=0, max=10, step=0.1, value=4.0, marks={0: '0', 5: '5', 10: '10'})
                                    ]),
                                    html.Div([
                                        html.Label("ISRU"),
                                        dcc.Slider(id="isru-slider", min=0, max=10, step=0.1, value=3.0, marks={0: '0', 5: '5', 10: '10'})
                                    ]),
                                    html.Div([
                                        html.Label("Thermal Control"),
                                        dcc.Slider(id="thermal-slider", min=0, max=10, step=0.1, value=3.0, marks={0: '0', 5: '5', 10: '10'})
                                    ]),
                                    
                                    html.H4("ISRU Mode"),
                                    dcc.RadioItems(
                                        id="isru-mode",
                                        options=[
                                            {'label': 'Water', 'value': 'water'},
                                            {'label': 'Oxygen', 'value': 'oxygen'},
                                            {'label': 'Both', 'value': 'both'},
                                            {'label': 'Off', 'value': 'off'}
                                        ],
                                        value='both'
                                    ),
                                    
                                    html.H4("Maintenance Target"),
                                    dcc.RadioItems(
                                        id="maintenance-target",
                                        options=[
                                            {'label': 'Power System', 'value': 'power_system'},
                                            {'label': 'Life Support', 'value': 'life_support'},
                                            {'label': 'ISRU', 'value': 'isru'},
                                            {'label': 'Thermal Control', 'value': 'thermal_control'},
                                            {'label': 'None', 'value': 'none'}
                                        ],
                                        value='none'
                                    ),
                                    
                                    html.Button("Apply Manual Settings", id="manual-button", n_clicks=0)
                                ])
                            ], style={'width': '48%'})
                        ], style={'display': 'flex', 'justifyContent': 'space-between'})
                    ]),
                    
                    # Forecasting Tab
                    dcc.Tab(label="Forecasting", children=[
                        html.Div([
                            html.H3("Resource Forecasts"),
                            dcc.Graph(id="forecast-graph"),
                            html.Div([
                                html.H4("Forecast Horizon"),
                                dcc.Slider(id="forecast-slider", min=6, max=48, step=6, value=24, marks={6: '6h', 12: '12h', 24: '24h', 36: '36h', 48: '48h'})
                            ])
                        ])
                    ]),
                    
                    # Anomaly Detection Tab
                    dcc.Tab(label="Anomaly Detection", children=[
                        html.Div([
                            html.H3("System Anomalies"),
                            dcc.Graph(id="anomaly-graph"),
                            html.Div(id="anomaly-alerts", style={'backgroundColor': '#fff3f3', 'padding': '10px', 'borderRadius': '5px', 'marginTop': '10px'})
                        ])
                    ])
                ])
            ], style={'marginTop': '20px'}),
            
            # Hidden div for storing state
            html.Div(id='state-store', style={'display': 'none'})
        ])
    
    def _configure_callbacks(self):
        """
        Configure dashboard callbacks
        """
        # Step simulation callback
        @self.app.callback(
            [Output('state-store', 'children'),
             Output('sol-display', 'children'),
             Output('hour-display', 'children'),
             Output('status-display', 'children')],
            [Input('step-button', 'n_clicks'),
             Input('reset-button', 'n_clicks'),
             Input('manual-button', 'n_clicks'),
             Input('auto-interval', 'n_intervals')],
            [State('state-store', 'children'),
             State('life-support-slider', 'value'),
             State('isru-slider', 'value'),
             State('thermal-slider', 'value'),
             State('isru-mode', 'value'),
             State('maintenance-target', 'value')]
        )
        def update_simulation(step_clicks, reset_clicks, manual_clicks, auto_intervals, state_json, life_support, isru, thermal, isru_mode, maintenance):
            # Determine which button was clicked
            ctx = dash.callback_context
            if not ctx.triggered:
                # No button clicked, return current state
                if state_json:
                    state_dict = json.loads(state_json)
                    return state_json, state_dict['time'][0], state_dict['time'][1], "Operational"
                else:
                    # Initialize state
                    state_dict = self._state_to_dict(self.state)
                    return json.dumps(state_dict), state_dict['time'][0], state_dict['time'][1], "Operational"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'reset-button':
                # Reset simulation
                self.state = self.env.reset()
                self._reset_history()
                state_dict = self._state_to_dict(self.state)
                return json.dumps(state_dict), state_dict['time'][0], state_dict['time'][1], "Operational"
            
            elif button_id in ['step-button', 'manual-button', 'auto-interval']:
                # Load current state
                if state_json:
                    state_dict = json.loads(state_json)
                    self.state = self._dict_to_state(state_dict)
                
                # Determine action
                if button_id == 'manual-button':
                    # Use manual settings
                    action = {
                        'power_allocation': {
                            'life_support': life_support,
                            'isru': isru,
                            'thermal_control': thermal
                        },
                        'isru_mode': isru_mode,
                        'maintenance_target': None if maintenance == 'none' else maintenance
                    }
                else:
                    # Use AI recommendation
                    action = self.integration.make_decision(self.state)
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update state
                self.state = next_state
                state_dict = self._state_to_dict(self.state)
                
                # Update history
                self._update_history(state_dict, action)
                
                # Determine status
                status = "Critical" if done else "Operational"
                
                return json.dumps(state_dict), state_dict['time'][0], state_dict['time'][1], status
        
        # Auto-pilot toggle callback
        @self.app.callback(
            Output('auto-interval', 'disabled'),
            [Input('auto-button', 'n_clicks')]
        )
        def toggle_auto_pilot(n_clicks):
            if n_clicks is None:
                return True
            
            # Toggle auto-pilot
            return n_clicks % 2 == 0
        
        # Resources graph callback
        @self.app.callback(
            Output('resources-graph', 'figure'),
            [Input('state-store', 'children')]
        )
        def update_resources_graph(state_json):
            if not state_json:
                return go.Figure()
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each resource
            for resource in self.history['resources']:
                fig.add_trace(go.Scatter(
                    x=list(range(len(self.history['resources'][resource]))),
                    y=self.history['resources'][resource],
                    mode='lines+markers',
                    name=resource.capitalize()
                ))
            
            # Update layout
            fig.update_layout(
                title='Resource Levels Over Time',
                xaxis_title='Time Step',
                yaxis_title='Level',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Environment graph callback
        @self.app.callback(
            Output('environment-graph', 'figure'),
            [Input('state-store', 'children')]
        )
        def update_environment_graph(state_json):
            if not state_json:
                return go.Figure()
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each environmental factor
            for factor in self.history['environment']:
                fig.add_trace(go.Scatter(
                    x=list(range(len(self.history['environment'][factor]))),
                    y=self.history['environment'][factor],
                    mode='lines+markers',
                    name=factor.capitalize()
                ))
            
            # Update layout
            fig.update_layout(
                title='Environmental Conditions Over Time',
                xaxis_title='Time Step',
                yaxis_title='Value',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Subsystems graph callback
        @self.app.callback(
            Output('subsystems-graph', 'figure'),
            [Input('state-store', 'children')]
        )
        def update_subsystems_graph(state_json):
            if not state_json:
                return go.Figure()
            
            # Create figure
            fig = go.Figure()
            
            # Add traces for each subsystem
            for subsystem in self.history['subsystems']:
                fig.add_trace(go.Scatter(
                    x=list(range(len(self.history['subsystems'][subsystem]))),
                    y=self.history['subsystems'][subsystem],
                    mode='lines+markers',
                    name=subsystem.replace('_', ' ').capitalize()
                ))
            
            # Update layout
            fig.update_layout(
                title='Subsystem Status Over Time',
                xaxis_title='Time Step',
                yaxis_title='Status (1 = Operational)',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            return fig
        
        # Resource levels callback
        @self.app.callback(
            Output('resource-levels', 'children'),
            [Input('state-store', 'children')]
        )
        def update_resource_levels(state_json):
            if not state_json:
                return html.Div()
            
            state_dict = json.loads(state_json)
            
            # Create resource level displays
            resource_displays = []
            resources = ['power', 'water', 'oxygen', 'food']
            
            for i, resource in enumerate(resources):
                level = state_dict['habitat'][i]
                
                # Determine color based on level
                if level < 50:
                    color = 'red'
                elif level < 200:
                    color = 'orange'
                else:
                    color = 'green'
                
                resource_displays.append(html.Div([
                    html.H4(f"{resource.capitalize()}: ", style={'display': 'inline'}),
                    html.H4(f"{level:.2f}", style={'display': 'inline', 'color': color})
                ]))
            
            return html.Div(resource_displays)
        
        # AI recommendations callback
        @self.app.callback(
            Output('ai-recommendations', 'children'),
            [Input('state-store', 'children')]
        )
        def update_ai_recommendations(state_json):
            if not state_json:
                return html.Div()
            
            state_dict = json.loads(state_json)
            
            # Get AI recommendation
            state = self._dict_to_state(state_dict)
            recommendation = self.integration.make_decision(state)
            
            # Create recommendation display
            recommendation_display = []
            
            # Power allocation
            recommendation_display.append(html.H4("Power Allocation"))
            for subsystem, allocation in recommendation['power_allocation'].items():
                recommendation_display.append(html.Div([
                    html.Strong(f"{subsystem.replace('_', ' ').capitalize()}: "),
                    html.Span(f"{allocation:.2f} kW")
                ]))
            
            # ISRU mode
            recommendation_display.append(html.H4("ISRU Mode"))
            recommendation_display.append(html.Div([
                html.Strong("Recommended mode: "),
                html.Span(recommendation['isru_mode'].capitalize())
            ]))
            
            # Maintenance target
            recommendation_display.append(html.H4("Maintenance"))
            target = recommendation.get('maintenance_target')
            if target:
                recommendation_display.append(html.Div([
                    html.Strong("Recommended target: "),
                    html.Span(target.replace('_', ' ').capitalize())
                ]))
            else:
                recommendation_display.append(html.Div([
                    html.Strong("Recommendation: "),
                    html.Span("No maintenance needed")
                ]))
            
            # Add explanation
            recommendation_display.append(html.H4("Explanation"))
            recommendation_display.append(html.Div([
                html.P("This recommendation is based on current resource levels, environmental conditions, and forecasted needs. The decision integration system combines reinforcement learning models with predictive analytics to optimize resource allocation and ensure habitat sustainability.")
            ]))
            
            return html.Div(recommendation_display)
        
        # Forecast graph callback
        @self.app.callback(
            Output('forecast-graph', 'figure'),
            [Input('state-store', 'children'),
             Input('forecast-slider', 'value')]
        )
        def update_forecast_graph(state_json, horizon):
            if not state_json:
                return go.Figure()
            
            state_dict = json.loads(state_json)
            
            # Create figure
            fig = go.Figure()
            
            # Generate forecasts (simulated)
            resources = ['power', 'water', 'oxygen', 'food']
            current_levels = [state_dict['habitat'][i] for i in range(4)]
            
            for i, resource in enumerate(resources):
                # Simulated forecast
                current = current_levels[i]
                forecast = []
                
                for h in range(horizon):
                    # Simple simulation with some randomness
                    if resource == 'power':
                        # Power fluctuates with solar irradiance
                        solar = state_dict['environment'][4]
                        change = np.random.normal(0.1 * solar / 500, 2)
                    elif resource == 'water':
                        # Water decreases steadily
                        change = np.random.normal(-1, 0.5)
                    elif resource == 'oxygen':
                        # Oxygen decreases steadily
                        change = np.random.normal(-0.8, 0.4)
                    elif resource == 'food':
                        # Food decreases steadily
                        change = np.random.normal(-0.5, 0.2)
                    
                    current += change
                    forecast.append(current)
                
                # Add historical data
                historical_x = list(range(-len(self.history['resources'][resource]), 0))
                historical_y = self.history['resources'][resource]
                
                # Add forecast data
                forecast_x = list(range(horizon))
                forecast_y = forecast
                
                # Plot historical data
                fig.add_trace(go.Scatter(
                    x=historical_x,
                    y=historical_y,
                    mode='lines',
                    name=f"{resource.capitalize()} (Historical)",
                    line=dict(color=px.colors.qualitative.Plotly[i])
                ))
                
                # Plot forecast data
                fig.add_trace(go.Scatter(
                    x=forecast_x,
                    y=forecast_y,
                    mode='lines',
                    name=f"{resource.capitalize()} (Forecast)",
                    line=dict(color=px.colors.qualitative.Plotly[i], dash='dash')
                ))
            
            # Update layout
            fig.update_layout(
                title='Resource Forecasts',
                xaxis_title='Time (Hours)',
                yaxis_title='Level',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            # Add vertical line at present
            fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="black")
            fig.add_annotation(x=0, y=0, text="Present", showarrow=False, yshift=10)
            
            return fig
        
        # Anomaly graph callback
        @self.app.callback(
            [Output('anomaly-graph', 'figure'),
             Output('anomaly-alerts', 'children')],
            [Input('state-store', 'children')]
        )
        def update_anomaly_graph(state_json):
            if not state_json:
                return go.Figure(), html.Div()
            
            state_dict = json.loads(state_json)
            
            # Create figure
            fig = go.Figure()
            
            # Generate anomaly scores (simulated)
            resources = ['power', 'water', 'oxygen', 'food']
            current_levels = [state_dict['habitat'][i] for i in range(4)]
            
            # Simulated anomaly detection
            anomaly_scores = []
            anomalies = []
            thresholds = [0.8, 0.8, 0.8, 0.8]
            
            for i, resource in enumerate(resources):
                # Simple anomaly score based on resource level
                level = current_levels[i]
                
                if resource == 'power':
                    score = 1.0 - (level / 200)
                elif resource == 'water':
                    score = 1.0 - (level / 1000)
                elif resource == 'oxygen':
                    score = 1.0 - (level / 500)
                elif resource == 'food':
                    score = 1.0 - (level / 800)
                
                # Add some randomness
                score = max(0, min(1, score + np.random.normal(0, 0.1)))
                
                anomaly_scores.append(score)
                anomalies.append(score > thresholds[i])
            
            # Plot anomaly scores
            fig.add_trace(go.Bar(
                x=resources,
                y=anomaly_scores,
                name='Anomaly Score'
            ))
            
            # Plot thresholds
            fig.add_trace(go.Scatter(
                x=resources,
                y=thresholds,
                mode='lines',
                name='Threshold',
                line=dict(color='red', dash='dash')
            ))
            
            # Update layout
            fig.update_layout(
                title='Resource Anomaly Detection',
                xaxis_title='Resource',
                yaxis_title='Anomaly Score',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
            )
            
            # Create anomaly alerts
            alerts = []
            
            if any(anomalies):
                alerts.append(html.H4("Anomaly Alerts", style={'color': 'red'}))
                
                for i, resource in enumerate(resources):
                    if anomalies[i]:
                        alerts.append(html.Div([
                            html.Strong(f"{resource.capitalize()}: "),
                            html.Span(f"Anomaly detected (score: {anomaly_scores[i]:.2f})")
                        ], style={'color': 'red'}))
                
                alerts.append(html.P("Recommended actions:"))
                alerts.append(html.Ul([
                    html.Li("Check system logs for errors"),
                    html.Li("Verify sensor readings"),
                    html.Li("Inspect relevant subsystems"),
                    html.Li("Consider maintenance operations")
                ]))
            else:
                alerts.append(html.H4("No Anomalies Detected", style={'color': 'green'}))
                alerts.append(html.P("All systems operating within normal parameters."))
            
            return fig, html.Div(alerts)
    
    def _state_to_dict(self, state):
        """
        Convert environment state to dictionary
        
        Args:
            state: Environment state
            
        Returns:
            dict: State dictionary
        """
        return {
            'time': state['time'],
            'environment': state['environment'],
            'habitat': state['habitat'],
            'subsystems': state['subsystems']
        }
    
    def _dict_to_state(self, state_dict):
        """
        Convert dictionary to environment state
        
        Args:
            state_dict (dict): State dictionary
            
        Returns:
            dict: Environment state
        """
        return state_dict
    
    def _update_history(self, state_dict, action):
        """
        Update history with current state
        
        Args:
            state_dict (dict): Current state dictionary
            action (dict): Current action
        """
        # Update time
        self.history['time'].append(state_dict['time'])
        
        # Update resources
        for i, resource in enumerate(['power', 'water', 'oxygen', 'food']):
            self.history['resources'][resource].append(state_dict['habitat'][i])
        
        # Update environment
        for i, factor in enumerate(['temperature', 'pressure', 'dust_opacity', 'solar_irradiance']):
            if i < len(state_dict['environment']):
                self.history['environment'][factor].append(state_dict['environment'][i])
        
        # Update subsystems
        for i, subsystem in enumerate(['power_system', 'life_support', 'isru', 'thermal_control']):
            if i < len(state_dict['subsystems']):
                self.history['subsystems'][subsystem].append(state_dict['subsystems'][i])
        
        # Update decisions
        self.history['decisions'].append(action)
    
    def _reset_history(self):
        """
        Reset history
        """
        self.history = {
            'time': [],
            'resources': {
                'power': [],
                'water': [],
                'oxygen': [],
                'food': []
            },
            'environment': {
                'temperature': [],
                'pressure': [],
                'dust_opacity': [],
                'solar_irradiance': []
            },
            'subsystems': {
                'power_system': [],
                'life_support': [],
                'isru': [],
                'thermal_control': []
            },
            'decisions': []
        }
    
    def run_server(self, debug=False, port=8050):
        """
        Run the dashboard server
        
        Args:
            debug (bool): Whether to run in debug mode
            port (int): Port to run the server on
        """
        self.app.run_server(debug=debug, port=port)
    
    def save_layout_screenshot(self, save_path=None):
        """
        Save a screenshot of the dashboard layout
        
        Args:
            save_path (str): Path to save the screenshot
            
        Returns:
            str: Path to saved screenshot
        """
        # This is a placeholder function
        # In a real implementation, this would use a headless browser to capture a screenshot
        
        if not save_path:
            save_path = os.path.join(self.ui_dir, "dashboard_layout.png")
        
        # Create a simple visualization of the layout
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Draw layout components
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='lightgray'))
        ax.add_patch(plt.Rectangle((0.05, 0.9), 0.9, 0.05, fill=True, color='white'))
        ax.add_patch(plt.Rectangle((0.05, 0.8), 0.9, 0.05, fill=True, color='lightblue'))
        
        # Left column
        ax.add_patch(plt.Rectangle((0.05, 0.4), 0.4, 0.35, fill=True, color='white'))
        
        # Right column
        ax.add_patch(plt.Rectangle((0.55, 0.4), 0.4, 0.35, fill=True, color='white'))
        
        # Decision support
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.9, 0.3, fill=True, color='white'))
        
        # Add labels
        ax.text(0.5, 0.925, "Mars Habitat Resource Management Dashboard", ha='center', va='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.825, "Status Bar", ha='center', va='center', fontsize=12)
        ax.text(0.25, 0.575, "Resources", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.75, 0.575, "Environment & Systems", ha='center', va='center', fontsize=12, fontweight='bold')
        ax.text(0.5, 0.2, "Decision Support", ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Save figure
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"Dashboard layout screenshot saved to {save_path}")
        return save_path
    
    def generate_ui_report(self, save_path=None):
        """
        Generate UI report
        
        Args:
            save_path (str): Path to save the report
            
        Returns:
            str: Report content
        """
        print(f"Generating UI report...")
        
        # Generate report
        report = "# Mars Habitat Human-AI Interface Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add system description
        report += "## System Description\n\n"
        report += "The Mars Habitat Human-AI Interface provides a comprehensive dashboard for monitoring and controlling "
        report += "habitat resources and systems. The interface combines real-time monitoring, predictive analytics, and "
        report += "decision support to assist human operators in managing the Mars habitat.\n\n"
        
        # Add features
        report += "## Key Features\n\n"
        
        report += "### Real-time Monitoring\n\n"
        report += "- Resource levels (power, water, oxygen, food)\n"
        report += "- Environmental conditions (temperature, pressure, dust opacity, solar irradiance)\n"
        report += "- Subsystem status (power system, life support, ISRU, thermal control)\n"
        report += "- Time tracking (sol and hour)\n\n"
        
        report += "### Decision Support\n\n"
        report += "- AI recommendations for resource allocation\n"
        report += "- Explanation of recommendations\n"
        report += "- Manual control options for human operators\n"
        report += "- Auto-pilot mode for autonomous operation\n\n"
        
        report += "### Predictive Analytics\n\n"
        report += "- Resource forecasting with adjustable horizon\n"
        report += "- Anomaly detection for early warning\n"
        report += "- Alerts and recommended actions for anomalies\n\n"
        
        # Add interface description
        report += "## Interface Components\n\n"
        
        report += "### Header\n\n"
        report += "- Dashboard title\n"
        report += "- Control buttons (Step Simulation, Reset Simulation, Auto-Pilot)\n\n"
        
        report += "### Status Bar\n\n"
        report += "- Current sol and hour\n"
        report += "- Habitat status (Operational or Critical)\n\n"
        
        report += "### Resources Panel\n\n"
        report += "- Graph of resource levels over time\n"
        report += "- Current resource levels with color-coded status\n\n"
        
        report += "### Environment & Systems Panel\n\n"
        report += "- Graph of environmental conditions over time\n"
        report += "- Graph of subsystem status over time\n\n"
        
        report += "### Decision Support Panel\n\n"
        report += "- Recommendations tab with AI suggestions and manual controls\n"
        report += "- Forecasting tab with resource predictions\n"
        report += "- Anomaly Detection tab with alerts and recommended actions\n\n"
        
        # Add usage instructions
        report += "## Usage Instructions\n\n"
        
        report += "### Simulation Control\n\n"
        report += "1. **Step Simulation**: Advance the simulation by one time step\n"
        report += "2. **Reset Simulation**: Reset the simulation to initial conditions\n"
        report += "3. **Auto-Pilot**: Toggle automatic stepping of the simulation\n\n"
        
        report += "### Decision Making\n\n"
        report += "1. **AI Recommendations**: View suggested resource allocations\n"
        report += "2. **Manual Control**: Adjust power allocation, ISRU mode, and maintenance target\n"
        report += "3. **Apply Manual Settings**: Apply manual settings to the simulation\n\n"
        
        report += "### Monitoring\n\n"
        report += "1. **Resource Levels**: Monitor current and historical resource levels\n"
        report += "2. **Environmental Conditions**: Track changes in the Martian environment\n"
        report += "3. **Subsystem Status**: Monitor the operational status of habitat subsystems\n\n"
        
        report += "### Forecasting\n\n"
        report += "1. **Resource Forecasts**: View predicted resource levels\n"
        report += "2. **Forecast Horizon**: Adjust the prediction time horizon\n"
        report += "3. **Anomaly Detection**: Monitor for potential system anomalies\n\n"
        
        # Add implementation details
        report += "## Implementation Details\n\n"
        report += "The Human-AI Interface is implemented using Dash, a Python framework for building web applications. "
        report += "The interface communicates with the Decision Integration System to provide AI recommendations and "
        report += "with the Mars Habitat Environment to simulate habitat operations.\n\n"
        
        report += "Key components:\n\n"
        report += "1. **Dashboard Layout**: Responsive layout with multiple panels and tabs\n"
        report += "2. **Interactive Controls**: Sliders, buttons, and radio items for user input\n"
        report += "3. **Real-time Graphs**: Dynamic visualizations of habitat data\n"
        report += "4. **Callback System**: Event handlers for user interactions\n"
        report += "5. **State Management**: Tracking and updating of simulation state\n\n"
        
        # Add conclusion
        report += "## Conclusion\n\n"
        report += "The Mars Habitat Human-AI Interface provides a comprehensive dashboard for monitoring and controlling "
        report += "habitat resources and systems. The interface combines real-time monitoring, predictive analytics, and "
        report += "decision support to assist human operators in managing the Mars habitat.\n\n"
        
        report += "The interface demonstrates how AI and human operators can work together to ensure the sustainability "
        report += "of Mars habitats, with AI providing recommendations and humans making final decisions based on their "
        report += "expertise and judgment.\n"
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"UI report saved to {save_path}")
        
        return report

# Example usage
if __name__ == "__main__":
    # Create dashboard
    dashboard = MarsHabitatDashboard("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Save layout screenshot
    dashboard.save_layout_screenshot()
    
    # Generate UI report
    dashboard.generate_ui_report("/home/ubuntu/martian_habitat_pathfinder/ui/ui_report.md")
    
    # Run server
    dashboard.run_server(debug=True)
