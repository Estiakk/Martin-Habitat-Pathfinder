# Simulation Environment Module

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random

class MarsHabitatSimulation:
    """
    Simulation environment for Mars habitat resource management:
    - Models Martian environmental conditions
    - Simulates habitat subsystems and their interactions
    - Provides interfaces for reinforcement learning agents
    - Supports scenario-based testing and evaluation
    """
    
    def __init__(self, data_dir, config=None):
        """
        Initialize the Mars habitat simulation environment
        
        Args:
            data_dir (str): Directory containing data and configuration files
            config (dict): Configuration parameters (optional)
        """
        self.data_dir = data_dir
        self.sim_dir = os.path.join(data_dir, "simulation")
        os.makedirs(self.sim_dir, exist_ok=True)
        
        # Load configuration or use defaults
        self.config = config if config else self._default_config()
        
        # Initialize simulation state
        self.reset()
        
        # Load environmental data if available
        self.env_data = self._load_environmental_data()
        
        print(f"Mars Habitat Simulation initialized with data directory: {data_dir}")
    
    def _default_config(self):
        """
        Create default configuration for simulation
        
        Returns:
            dict: Default configuration parameters
        """
        return {
            # Simulation parameters
            "sim_timestep": 1.0,  # hours
            "max_sol": 100,  # maximum simulation duration in sols
            "start_date": "2045-03-15",  # Mars mission start date
            
            # Environmental parameters
            "location": {
                "name": "Jezero Crater",
                "latitude": 18.4,
                "longitude": 77.7,
                "elevation": -2500,  # meters
            },
            "dust_storm_probability": 0.01,  # probability per sol
            "dust_storm_duration": {
                "min": 1,  # sols
                "max": 10  # sols
            },
            
            # Habitat parameters
            "habitat": {
                "modules": 3,
                "crew_size": 4,
                "pressurized_volume": 300,  # cubic meters
                "initial_resources": {
                    "power": 100,  # kWh in batteries
                    "water": 1000,  # liters
                    "oxygen": 500,  # kg
                    "food": 1000,  # kg
                    "spare_parts": 100  # units
                }
            },
            
            # Power system
            "power_system": {
                "solar_array_area": 50,  # square meters
                "solar_efficiency": 0.25,  # 25% efficiency
                "battery_capacity": 150,  # kWh
                "battery_efficiency": 0.9,
                "rtg_power": 1.0,  # kW (Radioisotope Thermoelectric Generator)
                "base_load": 5.0,  # kW (constant power draw)
            },
            
            # Life support system
            "life_support": {
                "oxygen_generation_rate": 1.0,  # kg per hour
                "water_recycling_efficiency": 0.85,
                "co2_scrubbing_capacity": 4.0,  # kg per hour
                "thermal_control_power": 2.0,  # kW
                "air_filtration_power": 0.5,  # kW
            },
            
            # ISRU (In-Situ Resource Utilization)
            "isru": {
                "enabled": True,
                "water_extraction_rate": 0.5,  # liters per hour
                "water_extraction_power": 2.0,  # kW
                "oxygen_production_rate": 0.2,  # kg per hour
                "oxygen_production_power": 3.0,  # kW
            },
            
            # Maintenance
            "maintenance": {
                "failure_rates": {
                    "power_system": 0.001,  # probability per hour
                    "life_support": 0.002,
                    "isru": 0.003,
                    "thermal_control": 0.001
                },
                "repair_times": {
                    "power_system": 4,  # hours
                    "life_support": 3,
                    "isru": 5,
                    "thermal_control": 2
                },
                "parts_required": {
                    "power_system": 2,  # units
                    "life_support": 1,
                    "isru": 3,
                    "thermal_control": 1
                }
            }
        }
    
    def _load_environmental_data(self):
        """
        Load environmental data from processed files if available
        
        Returns:
            dict: Environmental data or None if not available
        """
        env_data = {}
        
        # Try to load MEDA data
        meda_files = [f for f in os.listdir(os.path.join(self.data_dir, "processed")) 
                     if f.startswith("processed_meda_")]
        
        if meda_files:
            try:
                meda_df = pd.read_csv(os.path.join(self.data_dir, "processed", meda_files[0]))
                env_data["meda"] = meda_df
                print(f"Loaded MEDA environmental data: {len(meda_df)} records")
            except Exception as e:
                print(f"Warning: Could not load MEDA data: {e}")
        
        # Try to load terrain data
        terrain_files = [f for f in os.listdir(os.path.join(self.data_dir, "features")) 
                        if f.startswith("terrain_")]
        
        if terrain_files:
            try:
                terrain_df = pd.read_csv(os.path.join(self.data_dir, "features", terrain_files[0]))
                env_data["terrain"] = terrain_df
                print(f"Loaded terrain data: {len(terrain_df)} records")
            except Exception as e:
                print(f"Warning: Could not load terrain data: {e}")
        
        return env_data
    
    def reset(self):
        """
        Reset simulation to initial state
        
        Returns:
            dict: Initial observation
        """
        # Initialize time tracking
        self.current_sol = 0
        self.current_hour = 0
        self.total_hours = 0
        
        # Parse start date
        self.start_datetime = datetime.strptime(self.config["start_date"], "%Y-%m-%d")
        self.current_datetime = self.start_datetime
        
        # Initialize environmental conditions
        self.env_conditions = {
            "temperature": -60.0,  # Celsius
            "pressure": 730.0,  # Pascal
            "wind_speed": 0.0,  # m/s
            "dust_opacity": 0.5,  # dimensionless
            "solar_irradiance": 0.0,  # W/m^2
            "uv_index": 0.0,  # dimensionless
            "radiation_level": 0.0,  # mSv/day
            "is_dust_storm": False
        }
        
        # Initialize habitat state
        self.habitat_state = {
            "power": self.config["habitat"]["initial_resources"]["power"],
            "water": self.config["habitat"]["initial_resources"]["water"],
            "oxygen": self.config["habitat"]["initial_resources"]["oxygen"],
            "food": self.config["habitat"]["initial_resources"]["food"],
            "spare_parts": self.config["habitat"]["initial_resources"]["spare_parts"],
            "internal_temperature": 22.0,  # Celsius
            "internal_pressure": 101325.0,  # Pascal (Earth standard)
            "internal_humidity": 40.0,  # percent
            "co2_level": 0.1  # percent
        }
        
        # Initialize subsystem states
        self.subsystem_states = {
            "power_system": {
                "status": "operational",
                "solar_array_efficiency": self.config["power_system"]["solar_efficiency"],
                "battery_charge": self.config["habitat"]["initial_resources"]["power"],
                "power_generation": 0.0,
                "power_consumption": 0.0,
                "maintenance_needed": False
            },
            "life_support": {
                "status": "operational",
                "oxygen_generation": self.config["life_support"]["oxygen_generation_rate"],
                "water_recycling": self.config["life_support"]["water_recycling_efficiency"],
                "co2_scrubbing": self.config["life_support"]["co2_scrubbing_capacity"],
                "maintenance_needed": False
            },
            "isru": {
                "status": "operational" if self.config["isru"]["enabled"] else "disabled",
                "water_extraction": self.config["isru"]["water_extraction_rate"],
                "oxygen_production": self.config["isru"]["oxygen_production_rate"],
                "maintenance_needed": False
            },
            "thermal_control": {
                "status": "operational",
                "heating_power": 0.0,
                "cooling_power": 0.0,
                "maintenance_needed": False
            }
        }
        
        # Initialize crew state
        self.crew_state = {
            "size": self.config["habitat"]["crew_size"],
            "water_consumption": 2.5 * self.config["habitat"]["crew_size"],  # liters per hour
            "oxygen_consumption": 0.5 * self.config["habitat"]["crew_size"],  # kg per hour
            "food_consumption": 0.5 * self.config["habitat"]["crew_size"],  # kg per hour
            "activity_level": "normal"  # normal, high, low, emergency
        }
        
        # Initialize event log
        self.event_log = []
        
        # Initialize history tracking
        self.history = {
            "sol": [],
            "hour": [],
            "temperature": [],
            "solar_irradiance": [],
            "dust_opacity": [],
            "power_generation": [],
            "power_consumption": [],
            "battery_charge": [],
            "water_level": [],
            "oxygen_level": [],
            "food_level": [],
            "spare_parts": []
        }
        
        # Get initial observation
        observation = self._get_observation()
        
        # Log reset event
        self._log_event("simulation_reset", "Simulation reset to initial state")
        
        return observation
    
    def step(self, actions=None):
        """
        Advance simulation by one timestep
        
        Args:
            actions (dict): Actions to take (optional)
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Apply actions if provided
        if actions:
            self._apply_actions(actions)
        
        # Update environmental conditions
        self._update_environment()
        
        # Update subsystem states
        self._update_subsystems()
        
        # Update habitat state
        self._update_habitat()
        
        # Check for events (failures, dust storms, etc.)
        self._check_events()
        
        # Update time
        self._update_time()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if simulation is done
        done = self._is_done()
        
        # Additional info
        info = {
            "sol": self.current_sol,
            "hour": self.current_hour,
            "events": [e for e in self.event_log if e["timestamp"] == self.total_hours]
        }
        
        # Update history
        self._update_history()
        
        return observation, reward, done, info
    
    def _apply_actions(self, actions):
        """
        Apply actions to the simulation
        
        Args:
            actions (dict): Actions to take
        """
        # Example actions:
        # - power_allocation: dict of subsystem power allocations
        # - isru_mode: "water", "oxygen", "both", "off"
        # - maintenance_target: subsystem to perform maintenance on
        
        # Apply power allocation if provided
        if "power_allocation" in actions:
            for subsystem, power in actions["power_allocation"].items():
                if subsystem in self.subsystem_states:
                    # Ensure power allocation is within reasonable bounds
                    power = max(0, min(power, 10.0))  # Limit to 0-10 kW
                    
                    if subsystem == "life_support":
                        # Adjust life support parameters based on power
                        base_power = self.config["life_support"]["thermal_control_power"] + \
                                    self.config["life_support"]["air_filtration_power"]
                        
                        if power < base_power:
                            # Reduced power means reduced efficiency
                            efficiency_factor = power / base_power
                            self.subsystem_states["life_support"]["oxygen_generation"] = \
                                self.config["life_support"]["oxygen_generation_rate"] * efficiency_factor
                            self.subsystem_states["life_support"]["water_recycling"] = \
                                self.config["life_support"]["water_recycling_efficiency"] * efficiency_factor
                            self.subsystem_states["life_support"]["co2_scrubbing"] = \
                                self.config["life_support"]["co2_scrubbing_capacity"] * efficiency_factor
                        else:
                            # Normal or extra power
                            self.subsystem_states["life_support"]["oxygen_generation"] = \
                                self.config["life_support"]["oxygen_generation_rate"]
                            self.subsystem_states["life_support"]["water_recycling"] = \
                                self.config["life_support"]["water_recycling_efficiency"]
                            self.subsystem_states["life_support"]["co2_scrubbing"] = \
                                self.config["life_support"]["co2_scrubbing_capacity"]
                    
                    elif subsystem == "isru":
                        # Adjust ISRU parameters based on power
                        if power < 0.1:  # Effectively off
                            self.subsystem_states["isru"]["status"] = "disabled"
                            self.subsystem_states["isru"]["water_extraction"] = 0
                            self.subsystem_states["isru"]["oxygen_production"] = 0
                        else:
                            self.subsystem_states["isru"]["status"] = "operational"
                            # Scale extraction/production rates by power
                            water_power = min(power, self.config["isru"]["water_extraction_power"])
                            water_factor = water_power / self.config["isru"]["water_extraction_power"]
                            self.subsystem_states["isru"]["water_extraction"] = \
                                self.config["isru"]["water_extraction_rate"] * water_factor
                            
                            oxygen_power = max(0, power - water_power)
                            oxygen_factor = oxygen_power / self.config["isru"]["oxygen_production_power"]
                            self.subsystem_states["isru"]["oxygen_production"] = \
                                self.config["isru"]["oxygen_production_rate"] * oxygen_factor
                    
                    elif subsystem == "thermal_control":
                        # Adjust thermal control based on power
                        self.subsystem_states["thermal_control"]["heating_power"] = power
        
        # Apply ISRU mode if provided
        if "isru_mode" in actions:
            mode = actions["isru_mode"]
            
            if mode == "water":
                self.subsystem_states["isru"]["water_extraction"] = self.config["isru"]["water_extraction_rate"]
                self.subsystem_states["isru"]["oxygen_production"] = 0
            elif mode == "oxygen":
                self.subsystem_states["isru"]["water_extraction"] = 0
                self.subsystem_states["isru"]["oxygen_production"] = self.config["isru"]["oxygen_production_rate"]
            elif mode == "both":
                self.subsystem_states["isru"]["water_extraction"] = self.config["isru"]["water_extraction_rate"] * 0.5
                self.subsystem_states["isru"]["oxygen_production"] = self.config["isru"]["oxygen_production_rate"] * 0.5
            elif mode == "off":
                self.subsystem_states["isru"]["water_extraction"] = 0
                self.subsystem_states["isru"]["oxygen_production"] = 0
        
        # Apply maintenance action if provided
        if "maintenance_target" in actions:
            target = actions["maintenance_target"]
            
            if target in self.subsystem_states and self.habitat_state["spare_parts"] > 0:
                if self.subsystem_states[target]["maintenance_needed"]:
                    # Perform maintenance
                    parts_required = self.config["maintenance"]["parts_required"][target]
                    
                    if self.habitat_state["spare_parts"] >= parts_required:
                        self.habitat_state["spare_parts"] -= parts_required
                        self.subsystem_states[target]["maintenance_needed"] = False
                        self.subsystem_states[target]["status"] = "operational"
                        
                        # Log maintenance event
                        self._log_event(
                            "maintenance_performed", 
                            f"Maintenance performed on {target}. {parts_required} spare parts used."
                        )
                    else:
                        # Not enough spare parts
                        self._log_event(
                            "maintenance_failed", 
                            f"Maintenance on {target} failed. Insufficient spare parts."
                        )
                else:
                    # Preventive maintenance
                    parts_required = max(1, self.config["maintenance"]["parts_required"][target] // 2)
                    
                    if self.habitat_state["spare_parts"] >= parts_required:
                        self.habitat_state["spare_parts"] -= parts_required
                        
                        # Log preventive maintenance event
                        self._log_event(
                            "preventive_maintenance", 
                            f"Preventive maintenance performed on {target}. {parts_required} spare parts used."
                        )
                    else:
                        # Not enough spare parts
                        self._log_event(
                            "maintenance_failed", 
                            f"Preventive maintenance on {target} failed. Insufficient spare parts."
                        )
    
    def _update_environment(self):
        """
        Update environmental conditions based on time and location
        """
        # If we have real environmental data, use it
        if "meda" in self.env_data:
            meda_df = self.env_data["meda"]
            
            # Find closest time point in data
            if "sol" in meda_df.columns and "hour" in meda_df.columns:
                # Create time index in data
                meda_df["time_idx"] = meda_df["sol"] * 24 + meda_df["hour"]
                
                # Current time index
                current_idx = self.current_sol * 24 + self.current_hour
                
                # Find closest time point
                closest_idx = (meda_df["time_idx"] - current_idx).abs().idxmin()
                env_row = meda_df.loc[closest_idx]
                
                # Update environmental conditions from data
                self.env_conditions["temperature"] = env_row["temperature"] - 273.15 if "temperature" in env_row else self.env_conditions["temperature"]
                self.env_conditions["pressure"] = env_row["pressure"] if "pressure" in env_row else self.env_conditions["pressure"]
                self.env_conditions["wind_speed"] = env_row["wind_speed"] if "wind_speed" in env_row else self.env_conditions["wind_speed"]
                self.env_conditions["dust_opacity"] = env_row["dust_opacity"] if "dust_opacity" in env_row else self.env_conditions["dust_opacity"]
                self.env_conditions["uv_index"] = env_row["uv_radiation"] if "uv_radiation" in env_row else self.env_conditions["uv_index"]
                
                # Estimate solar irradiance from UV index
                self.env_conditions["solar_irradiance"] = env_row["uv_radiation"] * 100 if "uv_radiation" in env_row else self._calculate_solar_irradiance()
                
                return
        
        # Otherwise, use synthetic environmental model
        
        # Temperature varies with time of day
        # Martian day/night cycle with temperature range -120°C to 20°C
        hour_angle = 2 * np.pi * self.current_hour / 24
        base_temp = -50  # Base temperature
        temp_amplitude = 70  # Daily temperature swing
        self.env_conditions["temperature"] = base_temp + temp_amplitude * np.sin(hour_angle - np.pi/2)
        
        # Solar irradiance follows day/night cycle
        self.env_conditions["solar_irradiance"] = self._calculate_solar_irradiance()
        
        # Pressure varies slightly with temperature
        self.env_conditions["pressure"] = 730 + 20 * np.sin(hour_angle - np.pi/2)
        
        # Wind speed varies throughout the day
        self.env_conditions["wind_speed"] = 5 + 10 * np.abs(np.sin(hour_angle * 1.5))
        
        # Dust opacity increases during midday
        base_dust = 0.5
        dust_variation = 0.3
        self.env_conditions["dust_opacity"] = base_dust + dust_variation * np.sin(hour_angle)
        
        # Check for dust storm events
        if not self.env_conditions["is_dust_storm"]:
            # Random chance of dust storm starting
            if random.random() < self.config["dust_storm_probability"] / 24:  # Convert from per-sol to per-hour
                self.env_conditions["is_dust_storm"] = True
                self.env_conditions["dust_storm_duration"] = random.randint(
                    self.config["dust_storm_duration"]["min"] * 24,
                    self.config["dust_storm_duration"]["max"] * 24
                )
                self.env_conditions["dust_storm_intensity"] = random.uniform(0.7, 1.0)
                
                # Log dust storm event
                self._log_event(
                    "dust_storm_start", 
                    f"Dust storm started with intensity {self.env_conditions['dust_storm_intensity']:.2f}"
                )
        else:
            # Update dust storm duration
            self.env_conditions["dust_storm_duration"] -= 1
            
            # Check if dust storm has ended
            if self.env_conditions["dust_storm_duration"] <= 0:
                self.env_conditions["is_dust_storm"] = False
                
                # Log dust storm end event
                self._log_event("dust_storm_end", "Dust storm ended")
            else:
                # During dust storm, increase dust opacity
                self.env_conditions["dust_opacity"] = min(
                    0.9 + self.env_conditions["dust_storm_intensity"] * 0.1,
                    1.0
                )
                
                # Reduce solar irradiance due to dust
                self.env_conditions["solar_irradiance"] *= (1 - self.env_conditions["dust_opacity"])
                
                # Increase wind speed during dust storm
                self.env_conditions["wind_speed"] = 15 + 20 * self.env_conditions["dust_storm_intensity"]
        
        # UV index follows solar irradiance
        self.env_conditions["uv_index"] = max(0, self.env_conditions["solar_irradiance"] / 100)
        
        # Radiation level is higher on Mars due to thin atmosphere
        # Base level plus daily variation
        self.env_conditions["radiation_level"] = 0.5 + 0.2 * np.sin(hour_angle)  # mSv/day
    
    def _calculate_solar_irradiance(self):
        """
        Calculate solar irradiance based on time of day and dust opacity
        
        Returns:
            float: Solar irradiance in W/m^2
        """
        # Mars receives about 590 W/m^2 at its average distance from the Sun
        max_irradiance = 590
        
        # Calculate solar elevation angle
        hour_angle = 2 * np.pi * self.current_hour / 24
        
        # Simple day/night cycle
        if 6 <= self.current_hour < 18:
            # Daytime: follow sine curve
            elevation_angle = np.sin(np.pi * (self.current_hour - 6) / 12)
            irradiance = max_irradiance * elevation_angle
        else:
            # Nighttime
            irradiance = 0
        
        # Reduce irradiance due to dust opacity
        irradiance *= (1 - 0.8 * self.env_conditions["dust_opacity"])
        
        return max(0, irradiance)
    
    def _update_subsystems(self):
        """
        Update subsystem states based on environmental conditions and habitat state
        """
        # Update power system
        self._update_power_system()
        
        # Update life support system
        self._update_life_support()
        
        # Update ISRU system
        self._update_isru()
        
        # Update thermal control system
        self._update_thermal_control()
    
    def _update_power_system(self):
        """
        Update power system state
        """
        power_system = self.subsystem_states["power_system"]
        
        # Skip if system is not operational
        if power_system["status"] != "operational":
            power_system["power_generation"] = self.config["power_system"]["rtg_power"]
            return
        
        # Calculate solar power generation
        solar_irradiance = self.env_conditions["solar_irradiance"]
        solar_array_area = self.config["power_system"]["solar_array_area"]
        solar_efficiency = power_system["solar_array_efficiency"]
        
        # Reduce efficiency if solar arrays are dusty
        dust_factor = 1.0 - 0.5 * self.env_conditions["dust_opacity"]
        
        # Calculate solar power generation
        solar_power = solar_irradiance * solar_array_area * solar_efficiency * dust_factor / 1000  # kW
        
        # Add RTG (Radioisotope Thermoelectric Generator) power
        rtg_power = self.config["power_system"]["rtg_power"]
        
        # Total power generation
        power_system["power_generation"] = solar_power + rtg_power
        
        # Calculate power consumption
        base_load = self.config["power_system"]["base_load"]
        life_support_power = self.config["life_support"]["thermal_control_power"] + \
                            self.config["life_support"]["air_filtration_power"]
        
        isru_power = 0
        if self.subsystem_states["isru"]["status"] == "operational":
            water_extraction_power = self.config["isru"]["water_extraction_power"] * \
                                    (self.subsystem_states["isru"]["water_extraction"] / 
                                     self.config["isru"]["water_extraction_rate"])
            
            oxygen_production_power = self.config["isru"]["oxygen_production_power"] * \
                                    (self.subsystem_states["isru"]["oxygen_production"] / 
                                     self.config["isru"]["oxygen_production_rate"])
            
            isru_power = water_extraction_power + oxygen_production_power
        
        thermal_power = self.subsystem_states["thermal_control"]["heating_power"] + \
                       self.subsystem_states["thermal_control"]["cooling_power"]
        
        # Total power consumption
        power_consumption = base_load + life_support_power + isru_power + thermal_power
        power_system["power_consumption"] = power_consumption
        
        # Update battery charge
        net_power = power_system["power_generation"] - power_consumption
        
        if net_power > 0:
            # Charging battery
            battery_capacity = self.config["power_system"]["battery_capacity"]
            charge_efficiency = self.config["power_system"]["battery_efficiency"]
            
            # Add energy to battery with efficiency loss
            power_system["battery_charge"] = min(
                battery_capacity,
                power_system["battery_charge"] + net_power * charge_efficiency * self.config["sim_timestep"]
            )
        else:
            # Discharging battery
            power_system["battery_charge"] = max(
                0,
                power_system["battery_charge"] + net_power * self.config["sim_timestep"]
            )
            
            # Check if battery is depleted
            if power_system["battery_charge"] < 1.0:
                self._log_event("low_power", "Battery charge critically low")
        
        # Update habitat power level
        self.habitat_state["power"] = power_system["battery_charge"]
    
    def _update_life_support(self):
        """
        Update life support system state
        """
        life_support = self.subsystem_states["life_support"]
        
        # Skip if system is not operational
        if life_support["status"] != "operational":
            return
        
        # Oxygen generation
        oxygen_generation = life_support["oxygen_generation"] * self.config["sim_timestep"]
        self.habitat_state["oxygen"] += oxygen_generation
        
        # Water recycling
        water_consumed = self.crew_state["water_consumption"] * self.config["sim_timestep"]
        water_recycled = water_consumed * life_support["water_recycling"]
        self.habitat_state["water"] -= water_consumed - water_recycled
        
        # CO2 scrubbing affects internal air quality
        co2_produced = 0.5 * self.crew_state["oxygen_consumption"] * self.config["sim_timestep"]
        co2_scrubbed = min(co2_produced, life_support["co2_scrubbing"] * self.config["sim_timestep"])
        
        # Update CO2 level
        self.habitat_state["co2_level"] = max(
            0.04,  # Minimum CO2 level
            self.habitat_state["co2_level"] + (co2_produced - co2_scrubbed) / self.config["habitat"]["pressurized_volume"] * 100
        )
        
        # Check if CO2 level is too high
        if self.habitat_state["co2_level"] > 1.0:
            self._log_event("high_co2", f"CO2 level high: {self.habitat_state['co2_level']:.2f}%")
    
    def _update_isru(self):
        """
        Update ISRU (In-Situ Resource Utilization) system state
        """
        isru = self.subsystem_states["isru"]
        
        # Skip if system is not operational
        if isru["status"] != "operational":
            return
        
        # Water extraction
        water_extracted = isru["water_extraction"] * self.config["sim_timestep"]
        self.habitat_state["water"] += water_extracted
        
        # Oxygen production
        oxygen_produced = isru["oxygen_production"] * self.config["sim_timestep"]
        self.habitat_state["oxygen"] += oxygen_produced
    
    def _update_thermal_control(self):
        """
        Update thermal control system state
        """
        thermal_control = self.subsystem_states["thermal_control"]
        
        # Skip if system is not operational
        if thermal_control["status"] != "operational":
            return
        
        # Calculate heating/cooling needs based on external temperature
        external_temp = self.env_conditions["temperature"]
        internal_temp = self.habitat_state["internal_temperature"]
        target_temp = 22.0  # Target internal temperature
        
        # Temperature difference affects power needs
        temp_diff = target_temp - external_temp
        
        if temp_diff > 0:
            # Need heating
            thermal_control["heating_power"] = 0.1 * temp_diff
            thermal_control["cooling_power"] = 0.0
        else:
            # Need cooling
            thermal_control["heating_power"] = 0.0
            thermal_control["cooling_power"] = 0.05 * abs(temp_diff)
        
        # Update internal temperature based on thermal control effectiveness
        # If power system can provide enough power, maintain target temperature
        # Otherwise, internal temperature will drift toward external temperature
        
        power_available = self.subsystem_states["power_system"]["battery_charge"] > 10.0
        
        if power_available:
            # Effective thermal control
            self.habitat_state["internal_temperature"] = target_temp + random.uniform(-0.5, 0.5)
        else:
            # Ineffective thermal control, temperature drifts toward external
            drift_rate = 0.1  # Temperature change per hour
            self.habitat_state["internal_temperature"] += (external_temp - internal_temp) * drift_rate
            
            # Log temperature warning if too far from target
            if abs(self.habitat_state["internal_temperature"] - target_temp) > 5.0:
                self._log_event(
                    "temperature_warning", 
                    f"Internal temperature outside comfort range: {self.habitat_state['internal_temperature']:.1f}°C"
                )
    
    def _update_habitat(self):
        """
        Update overall habitat state
        """
        # Update resource consumption by crew
        self._update_crew_consumption()
        
        # Check resource levels
        self._check_resource_levels()
    
    def _update_crew_consumption(self):
        """
        Update resource consumption by crew
        """
        # Oxygen consumption
        oxygen_consumed = self.crew_state["oxygen_consumption"] * self.config["sim_timestep"]
        self.habitat_state["oxygen"] = max(0, self.habitat_state["oxygen"] - oxygen_consumed)
        
        # Water consumption is handled in life support update
        
        # Food consumption
        food_consumed = self.crew_state["food_consumption"] * self.config["sim_timestep"]
        self.habitat_state["food"] = max(0, self.habitat_state["food"] - food_consumed)
    
    def _check_resource_levels(self):
        """
        Check resource levels and log warnings if low
        """
        # Check oxygen level
        if self.habitat_state["oxygen"] < self.crew_state["oxygen_consumption"] * 24:
            self._log_event("low_oxygen", f"Oxygen level critical: {self.habitat_state['oxygen']:.1f} kg")
        
        # Check water level
        if self.habitat_state["water"] < self.crew_state["water_consumption"] * 24:
            self._log_event("low_water", f"Water level critical: {self.habitat_state['water']:.1f} liters")
        
        # Check food level
        if self.habitat_state["food"] < self.crew_state["food_consumption"] * 24:
            self._log_event("low_food", f"Food level critical: {self.habitat_state['food']:.1f} kg")
        
        # Check spare parts
        if self.habitat_state["spare_parts"] < 5:
            self._log_event("low_spare_parts", f"Spare parts low: {self.habitat_state['spare_parts']} units")
    
    def _check_events(self):
        """
        Check for random events like equipment failures
        """
        # Check for subsystem failures
        for subsystem, state in self.subsystem_states.items():
            if state["status"] == "operational" and not state["maintenance_needed"]:
                # Check for random failure
                failure_rate = self.config["maintenance"]["failure_rates"].get(subsystem, 0.001)
                
                # Increase failure rate if system is under stress
                if subsystem == "power_system" and self.env_conditions["is_dust_storm"]:
                    failure_rate *= 2.0
                
                if subsystem == "thermal_control" and abs(self.habitat_state["internal_temperature"] - 22.0) > 10.0:
                    failure_rate *= 2.0
                
                # Random failure check
                if random.random() < failure_rate:
                    state["maintenance_needed"] = True
                    
                    # Log failure event
                    self._log_event(
                        "subsystem_failure", 
                        f"{subsystem} requires maintenance"
                    )
                    
                    # If not maintained promptly, system will fail
                    failure_countdown = self.config["maintenance"]["repair_times"].get(subsystem, 4)
                    state["failure_countdown"] = failure_countdown
            
            # Check if maintenance-needed system will fail
            if state["maintenance_needed"] and state["status"] == "operational":
                if "failure_countdown" in state:
                    state["failure_countdown"] -= 1
                    
                    if state["failure_countdown"] <= 0:
                        state["status"] = "failed"
                        
                        # Log system failure event
                        self._log_event(
                            "system_failure", 
                            f"{subsystem} has failed due to lack of maintenance"
                        )
    
    def _update_time(self):
        """
        Update simulation time
        """
        # Increment hour
        self.current_hour += self.config["sim_timestep"]
        self.total_hours += self.config["sim_timestep"]
        
        # Check for sol rollover
        if self.current_hour >= 24:
            self.current_hour = 0
            self.current_sol += 1
            
            # Log new sol
            self._log_event("new_sol", f"Sol {self.current_sol} started")
        
        # Update datetime
        # Mars sol is about 24 hours and 40 minutes
        # For simplicity, we'll use Earth hours but track Mars sols
        self.current_datetime = self.start_datetime + timedelta(hours=self.total_hours)
    
    def _get_observation(self):
        """
        Get current observation of simulation state
        
        Returns:
            dict: Current observation
        """
        observation = {
            "time": {
                "sol": self.current_sol,
                "hour": self.current_hour,
                "total_hours": self.total_hours
            },
            "environment": self.env_conditions.copy(),
            "habitat": self.habitat_state.copy(),
            "subsystems": {
                name: state.copy() for name, state in self.subsystem_states.items()
            },
            "crew": self.crew_state.copy()
        }
        
        return observation
    
    def _calculate_reward(self):
        """
        Calculate reward based on current state
        
        Returns:
            float: Reward value
        """
        reward = 0.0
        
        # Reward for maintaining resource levels
        # Normalize resource levels to 0-1 range
        power_level = min(1.0, self.habitat_state["power"] / self.config["power_system"]["battery_capacity"])
        water_level = min(1.0, self.habitat_state["water"] / self.config["habitat"]["initial_resources"]["water"])
        oxygen_level = min(1.0, self.habitat_state["oxygen"] / self.config["habitat"]["initial_resources"]["oxygen"])
        food_level = min(1.0, self.habitat_state["food"] / self.config["habitat"]["initial_resources"]["food"])
        
        # Resource reward: higher levels are better, but diminishing returns
        # Using sqrt to reward maintaining moderate levels more than stockpiling
        resource_reward = (np.sqrt(power_level) + np.sqrt(water_level) + 
                          np.sqrt(oxygen_level) + np.sqrt(food_level)) / 4.0
        
        # Penalty for critical resource levels
        critical_penalty = 0.0
        if power_level < 0.1:
            critical_penalty += 0.5
        if water_level < 0.1:
            critical_penalty += 0.5
        if oxygen_level < 0.1:
            critical_penalty += 1.0
        if food_level < 0.1:
            critical_penalty += 0.3
        
        # Reward for system health
        system_health_reward = 0.0
        for name, state in self.subsystem_states.items():
            if state["status"] == "operational":
                system_health_reward += 0.25
            elif state["status"] == "failed":
                system_health_reward -= 0.5
        
        # Normalize system health reward
        system_health_reward /= len(self.subsystem_states)
        
        # Reward for comfortable habitat conditions
        comfort_reward = 0.0
        
        # Temperature comfort (22°C is ideal)
        temp_diff = abs(self.habitat_state["internal_temperature"] - 22.0)
        if temp_diff < 3.0:
            comfort_reward += 0.1
        elif temp_diff > 10.0:
            comfort_reward -= 0.2
        
        # CO2 level comfort
        if self.habitat_state["co2_level"] < 0.5:
            comfort_reward += 0.1
        elif self.habitat_state["co2_level"] > 1.0:
            comfort_reward -= 0.3
        
        # Combine rewards
        reward = 0.4 * resource_reward + 0.3 * system_health_reward + 0.1 * comfort_reward - critical_penalty
        
        # Scale reward to reasonable range
        reward = max(-1.0, min(1.0, reward))
        
        return reward
    
    def _is_done(self):
        """
        Check if simulation is done
        
        Returns:
            bool: True if simulation is done
        """
        # Check if maximum sol reached
        if self.current_sol >= self.config["max_sol"]:
            return True
        
        # Check for critical failure conditions
        
        # No oxygen
        if self.habitat_state["oxygen"] <= 0:
            self._log_event("critical_failure", "No oxygen remaining")
            return True
        
        # No water
        if self.habitat_state["water"] <= 0:
            self._log_event("critical_failure", "No water remaining")
            return True
        
        # No food
        if self.habitat_state["food"] <= 0:
            self._log_event("critical_failure", "No food remaining")
            return True
        
        # Extreme temperature
        if self.habitat_state["internal_temperature"] < -10 or self.habitat_state["internal_temperature"] > 40:
            self._log_event("critical_failure", f"Extreme internal temperature: {self.habitat_state['internal_temperature']:.1f}°C")
            return True
        
        # Extreme CO2 level
        if self.habitat_state["co2_level"] > 5.0:
            self._log_event("critical_failure", f"Extreme CO2 level: {self.habitat_state['co2_level']:.1f}%")
            return True
        
        return False
    
    def _log_event(self, event_type, description):
        """
        Log an event in the simulation
        
        Args:
            event_type (str): Type of event
            description (str): Event description
        """
        event = {
            "timestamp": self.total_hours,
            "sol": self.current_sol,
            "hour": self.current_hour,
            "datetime": self.current_datetime.strftime("%Y-%m-%d %H:%M"),
            "type": event_type,
            "description": description
        }
        
        self.event_log.append(event)
    
    def _update_history(self):
        """
        Update history tracking
        """
        self.history["sol"].append(self.current_sol)
        self.history["hour"].append(self.current_hour)
        self.history["temperature"].append(self.env_conditions["temperature"])
        self.history["solar_irradiance"].append(self.env_conditions["solar_irradiance"])
        self.history["dust_opacity"].append(self.env_conditions["dust_opacity"])
        self.history["power_generation"].append(self.subsystem_states["power_system"]["power_generation"])
        self.history["power_consumption"].append(self.subsystem_states["power_system"]["power_consumption"])
        self.history["battery_charge"].append(self.subsystem_states["power_system"]["battery_charge"])
        self.history["water_level"].append(self.habitat_state["water"])
        self.history["oxygen_level"].append(self.habitat_state["oxygen"])
        self.history["food_level"].append(self.habitat_state["food"])
        self.history["spare_parts"].append(self.habitat_state["spare_parts"])
    
    def run_simulation(self, num_steps=24, actions_callback=None):
        """
        Run simulation for specified number of steps
        
        Args:
            num_steps (int): Number of steps to run
            actions_callback (callable): Function to generate actions at each step
            
        Returns:
            dict: Simulation results
        """
        observations = []
        rewards = []
        events = []
        
        for _ in range(num_steps):
            # Get actions if callback provided
            actions = None
            if actions_callback:
                actions = actions_callback(self._get_observation())
            
            # Step simulation
            observation, reward, done, info = self.step(actions)
            
            # Record results
            observations.append(observation)
            rewards.append(reward)
            events.extend(info["events"])
            
            # Check if simulation is done
            if done:
                break
        
        # Compile results
        results = {
            "observations": observations,
            "rewards": rewards,
            "events": events,
            "history": self.history,
            "final_state": self._get_observation()
        }
        
        return results
    
    def visualize_history(self, save_path=None):
        """
        Visualize simulation history
        
        Args:
            save_path (str): Path to save visualization (optional)
            
        Returns:
            tuple: Figure and axes objects
        """
        # Create figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        
        # Create time axis
        time = [self.history["sol"][i] + self.history["hour"][i]/24 for i in range(len(self.history["sol"]))]
        
        # Plot environmental conditions
        axs[0, 0].plot(time, self.history["temperature"], 'r-', label='Temperature (°C)')
        axs[0, 0].set_ylabel('Temperature (°C)')
        ax2 = axs[0, 0].twinx()
        ax2.plot(time, self.history["dust_opacity"], 'b-', label='Dust Opacity')
        ax2.set_ylabel('Dust Opacity')
        axs[0, 0].set_title('Environmental Conditions')
        axs[0, 0].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Plot power system
        axs[0, 1].plot(time, self.history["solar_irradiance"], 'y-', label='Solar Irradiance (W/m²)')
        axs[0, 1].set_ylabel('Solar Irradiance (W/m²)')
        ax2 = axs[0, 1].twinx()
        ax2.plot(time, self.history["power_generation"], 'g-', label='Power Generation (kW)')
        ax2.plot(time, self.history["power_consumption"], 'r-', label='Power Consumption (kW)')
        ax2.set_ylabel('Power (kW)')
        axs[0, 1].set_title('Power System')
        axs[0, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Plot battery charge
        axs[1, 0].plot(time, self.history["battery_charge"], 'b-', label='Battery Charge (kWh)')
        axs[1, 0].set_ylabel('Battery Charge (kWh)')
        axs[1, 0].set_title('Battery Status')
        axs[1, 0].legend()
        
        # Plot water level
        axs[1, 1].plot(time, self.history["water_level"], 'c-', label='Water (L)')
        axs[1, 1].set_ylabel('Water (L)')
        axs[1, 1].set_title('Water Level')
        axs[1, 1].legend()
        
        # Plot oxygen level
        axs[2, 0].plot(time, self.history["oxygen_level"], 'g-', label='Oxygen (kg)')
        axs[2, 0].set_ylabel('Oxygen (kg)')
        axs[2, 0].set_title('Oxygen Level')
        axs[2, 0].legend()
        
        # Plot food and spare parts
        axs[2, 1].plot(time, self.history["food_level"], 'y-', label='Food (kg)')
        axs[2, 1].set_ylabel('Food (kg)')
        ax2 = axs[2, 1].twinx()
        ax2.plot(time, self.history["spare_parts"], 'm-', label='Spare Parts')
        ax2.set_ylabel('Spare Parts')
        axs[2, 1].set_title('Food and Spare Parts')
        axs[2, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        # Set common x-axis label
        for ax in axs.flat:
            ax.set_xlabel('Sol')
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Visualization saved to {save_path}")
        
        return fig, axs
    
    def save_simulation_data(self, file_path):
        """
        Save simulation data to file
        
        Args:
            file_path (str): Path to save file
            
        Returns:
            str: Path to saved file
        """
        # Compile data to save
        data = {
            "config": self.config,
            "history": self.history,
            "events": self.event_log,
            "final_state": self._get_observation()
        }
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Simulation data saved to {file_path}")
        return file_path

# Example usage
if __name__ == "__main__":
    sim = MarsHabitatSimulation("/home/ubuntu/martian_habitat_pathfinder/data")
    
    # Run simulation for 24 hours (1 sol)
    results = sim.run_simulation(24)
    
    # Visualize results
    sim.visualize_history("/home/ubuntu/martian_habitat_pathfinder/simulations/simulation_results.png")
    
    # Save simulation data
    sim.save_simulation_data("/home/ubuntu/martian_habitat_pathfinder/simulations/simulation_data.json")
