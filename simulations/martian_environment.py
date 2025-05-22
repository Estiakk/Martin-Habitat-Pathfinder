# Martian Environment Model

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class MartianEnvironmentModel:
    """
    Model of Martian environmental conditions:
    - Temperature cycles (diurnal and seasonal)
    - Solar radiation patterns
    - Dust storm generation and effects
    - Atmospheric pressure variations
    - Wind patterns
    """
    
    def __init__(self, location=None):
        """
        Initialize the Martian environment model
        
        Args:
            location (dict): Location parameters (latitude, longitude, elevation)
        """
        # Default location (Jezero Crater)
        self.location = location if location else {
            "name": "Jezero Crater",
            "latitude": 18.4,
            "longitude": 77.7,
            "elevation": -2500  # meters
        }
        
        # Mars orbital parameters
        self.mars_year_length = 687  # Earth days
        self.mars_day_length = 24.6  # hours
        self.mars_axial_tilt = 25.19  # degrees
        self.mars_orbital_eccentricity = 0.0934
        
        # Mars seasons (Ls values in degrees)
        # Ls = 0: Northern Spring Equinox
        # Ls = 90: Northern Summer Solstice
        # Ls = 180: Northern Fall Equinox
        # Ls = 270: Northern Winter Solstice
        self.seasons = {
            "northern_spring": (0, 90),
            "northern_summer": (90, 180),
            "northern_fall": (180, 270),
            "northern_winter": (270, 360)
        }
        
        # Dust storm parameters
        self.dust_storm_season_factor = {
            "northern_spring": 0.5,
            "northern_summer": 1.0,  # Peak dust storm season
            "northern_fall": 2.0,    # Highest dust storm probability
            "northern_winter": 1.0
        }
        
        # Initialize state
        self.reset()
        
        print(f"Martian Environment Model initialized for {self.location['name']}")
    
    def reset(self):
        """
        Reset environment to initial state
        """
        # Time tracking
        self.current_sol = 0
        self.current_hour = 0
        self.current_ls = 0  # Solar longitude (season indicator)
        
        # Environmental state
        self.temperature = -60.0  # Celsius
        self.pressure = 730.0  # Pascal
        self.wind_speed = 0.0  # m/s
        self.wind_direction = 0.0  # degrees
        self.dust_opacity = 0.5  # dimensionless (0-1)
        self.solar_irradiance = 0.0  # W/m^2
        self.uv_index = 0.0  # dimensionless
        self.radiation_level = 0.0  # mSv/day
        
        # Dust storm state
        self.dust_storm_active = False
        self.dust_storm_duration = 0
        self.dust_storm_intensity = 0.0
        
        # History tracking
        self.history = {
            "sol": [],
            "hour": [],
            "ls": [],
            "temperature": [],
            "pressure": [],
            "wind_speed": [],
            "dust_opacity": [],
            "solar_irradiance": []
        }
    
    def get_current_season(self):
        """
        Get current Martian season based on Ls value
        
        Returns:
            str: Current season name
        """
        for season, (start, end) in self.seasons.items():
            if start <= self.current_ls < end:
                return season
        return "northern_winter"  # Default if something goes wrong
    
    def update(self, hours=1.0):
        """
        Update environment for specified time period
        
        Args:
            hours (float): Hours to advance
            
        Returns:
            dict: Current environmental conditions
        """
        # Update time
        self.current_hour += hours
        
        # Check for sol rollover
        if self.current_hour >= 24:
            self.current_sol += int(self.current_hour / 24)
            self.current_hour = self.current_hour % 24
            
            # Update Ls (seasonal progress)
            # One sol is approximately 0.5 degrees of Ls
            self.current_ls = (self.current_ls + 0.5 * int(hours / 24)) % 360
        
        # Get current season
        current_season = self.get_current_season()
        
        # Update temperature based on time of day and season
        self._update_temperature(current_season)
        
        # Update solar irradiance
        self._update_solar_irradiance(current_season)
        
        # Update atmospheric conditions
        self._update_atmospheric_conditions(current_season)
        
        # Check for dust storm events
        self._check_dust_storm_events(current_season)
        
        # Update history
        self._update_history()
        
        # Return current conditions
        return self.get_conditions()
    
    def _update_temperature(self, season):
        """
        Update temperature based on time of day and season
        
        Args:
            season (str): Current Martian season
        """
        # Base temperature varies by season
        season_base_temp = {
            "northern_spring": -60,
            "northern_summer": -40,
            "northern_fall": -60,
            "northern_winter": -80
        }
        
        # Temperature amplitude varies by season
        season_temp_amplitude = {
            "northern_spring": 60,
            "northern_summer": 70,
            "northern_fall": 60,
            "northern_winter": 50
        }
        
        # Get base temperature and amplitude for current season
        base_temp = season_base_temp.get(season, -60)
        temp_amplitude = season_temp_amplitude.get(season, 60)
        
        # Adjust for latitude (colder at poles, warmer at equator)
        latitude_factor = 1.0 - abs(self.location["latitude"]) / 90 * 0.5
        base_temp = base_temp * (0.8 + 0.4 * latitude_factor)
        
        # Adjust for elevation (colder at higher elevations)
        # Mars has a less steep lapse rate than Earth
        elevation_factor = max(0, -self.location["elevation"] / 10000)  # Normalize to 0-1 range
        base_temp = base_temp + 10 * elevation_factor  # Up to 10°C warmer at lower elevations
        
        # Calculate diurnal temperature variation
        hour_angle = 2 * np.pi * self.current_hour / 24
        diurnal_variation = temp_amplitude * np.sin(hour_angle - np.pi/2)
        
        # Calculate temperature
        self.temperature = base_temp + diurnal_variation * latitude_factor
        
        # Adjust for dust storms
        if self.dust_storm_active:
            # Dust storms reduce temperature swings
            dust_damping = 0.5 * self.dust_storm_intensity
            self.temperature = base_temp + diurnal_variation * (1 - dust_damping)
    
    def _update_solar_irradiance(self, season):
        """
        Update solar irradiance based on time of day, season, and dust conditions
        
        Args:
            season (str): Current Martian season
        """
        # Mars receives about 590 W/m^2 at its average distance from the Sun
        # This varies with orbital position (season)
        season_irradiance_factor = {
            "northern_spring": 0.95,  # Mars moving away from Sun
            "northern_summer": 0.85,  # Mars farthest from Sun (aphelion)
            "northern_fall": 0.95,   # Mars moving toward Sun
            "northern_winter": 1.15   # Mars closest to Sun (perihelion)
        }
        
        max_irradiance = 590 * season_irradiance_factor.get(season, 1.0)
        
        # Calculate solar elevation angle
        hour_angle = 2 * np.pi * self.current_hour / 24
        
        # Simple day/night cycle
        if 6 <= self.current_hour < 18:
            # Daytime: follow sine curve
            elevation_angle = np.sin(np.pi * (self.current_hour - 6) / 12)
            
            # Adjust for latitude and season (solar declination)
            # This is a simplified model of solar declination
            declination = self.mars_axial_tilt * np.sin(np.radians(self.current_ls))
            latitude_rad = np.radians(self.location["latitude"])
            
            # Adjust elevation angle based on latitude and declination
            elevation_factor = np.cos(latitude_rad - np.radians(declination))
            elevation_angle *= elevation_factor
            
            irradiance = max_irradiance * max(0, elevation_angle)
        else:
            # Nighttime
            irradiance = 0
        
        # Reduce irradiance due to dust opacity
        irradiance *= (1 - 0.8 * self.dust_opacity)
        
        self.solar_irradiance = max(0, irradiance)
        
        # Update UV index based on solar irradiance
        # Mars has less atmospheric protection from UV
        self.uv_index = self.solar_irradiance / 50  # Higher UV per unit of solar energy than Earth
        
        # Update radiation level
        # Base radiation level is higher on Mars due to thin atmosphere
        self.radiation_level = 0.5 + 0.2 * elevation_angle  # mSv/day
    
    def _update_atmospheric_conditions(self, season):
        """
        Update atmospheric conditions based on time of day and season
        
        Args:
            season (str): Current Martian season
        """
        # Pressure varies with season due to CO2 cycle
        season_pressure_factor = {
            "northern_spring": 1.0,
            "northern_summer": 0.9,  # CO2 frozen at south pole
            "northern_fall": 1.0,
            "northern_winter": 1.1   # CO2 frozen at north pole
        }
        
        # Base pressure at datum level
        base_pressure = 730 * season_pressure_factor.get(season, 1.0)
        
        # Adjust for elevation using barometric formula
        # Mars has a larger scale height than Earth
        scale_height = 11000  # meters
        elevation = self.location["elevation"]
        self.pressure = base_pressure * np.exp(-elevation / scale_height)
        
        # Add small diurnal variation
        hour_angle = 2 * np.pi * self.current_hour / 24
        self.pressure += 10 * np.sin(hour_angle)
        
        # Wind speed varies with time of day and season
        # Stronger winds during seasonal transitions
        season_wind_factor = {
            "northern_spring": 1.2,
            "northern_summer": 0.8,
            "northern_fall": 1.2,
            "northern_winter": 1.0
        }
        
        # Base wind speed
        base_wind = 5.0 * season_wind_factor.get(season, 1.0)
        
        # Diurnal variation (stronger during day-night transitions)
        time_factor = np.sin(2 * hour_angle)
        self.wind_speed = base_wind + 10 * abs(time_factor)
        
        # Wind direction changes throughout the day
        self.wind_direction = (self.current_hour * 15 + 180) % 360
        
        # Dust opacity has seasonal patterns
        season_dust_factor = {
            "northern_spring": 0.3,
            "northern_summer": 0.5,
            "northern_fall": 0.7,
            "northern_winter": 0.4
        }
        
        # Base dust opacity
        base_dust = 0.2 + 0.3 * season_dust_factor.get(season, 0.5)
        
        # Diurnal variation (more dust during day due to thermal currents)
        if 8 <= self.current_hour < 16:
            dust_variation = 0.2 * np.sin(np.pi * (self.current_hour - 8) / 8)
        else:
            dust_variation = 0
        
        # Update dust opacity
        if not self.dust_storm_active:
            self.dust_opacity = min(1.0, base_dust + dust_variation)
        else:
            # During dust storm, opacity is much higher
            self.dust_opacity = min(1.0, 0.7 + 0.3 * self.dust_storm_intensity)
    
    def _check_dust_storm_events(self, season):
        """
        Check for dust storm events
        
        Args:
            season (str): Current Martian season
        """
        # If dust storm is already active, update its duration
        if self.dust_storm_active:
            self.dust_storm_duration -= 1
            
            # Check if dust storm has ended
            if self.dust_storm_duration <= 0:
                self.dust_storm_active = False
                self.dust_storm_intensity = 0.0
                return
        
        # Check for new dust storm
        # Base probability per hour
        base_probability = 0.0001
        
        # Adjust for season
        season_factor = self.dust_storm_season_factor.get(season, 1.0)
        
        # Adjust for latitude (more dust storms in southern hemisphere during summer)
        latitude_factor = 1.0
        if season == "northern_summer" and self.location["latitude"] < 0:
            latitude_factor = 2.0
        
        # Calculate probability
        probability = base_probability * season_factor * latitude_factor
        
        # Random check for dust storm
        if np.random.random() < probability:
            self.dust_storm_active = True
            self.dust_storm_intensity = np.random.uniform(0.7, 1.0)
            
            # Duration depends on intensity (stronger storms last longer)
            min_duration = 24  # hours
            max_duration = 240  # hours (10 sols)
            self.dust_storm_duration = int(min_duration + (max_duration - min_duration) * self.dust_storm_intensity)
    
    def _update_history(self):
        """
        Update history tracking
        """
        self.history["sol"].append(self.current_sol)
        self.history["hour"].append(self.current_hour)
        self.history["ls"].append(self.current_ls)
        self.history["temperature"].append(self.temperature)
        self.history["pressure"].append(self.pressure)
        self.history["wind_speed"].append(self.wind_speed)
        self.history["dust_opacity"].append(self.dust_opacity)
        self.history["solar_irradiance"].append(self.solar_irradiance)
    
    def get_conditions(self):
        """
        Get current environmental conditions
        
        Returns:
            dict: Current conditions
        """
        return {
            "time": {
                "sol": self.current_sol,
                "hour": self.current_hour,
                "ls": self.current_ls,
                "season": self.get_current_season()
            },
            "temperature": self.temperature,
            "pressure": self.pressure,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction,
            "dust_opacity": self.dust_opacity,
            "solar_irradiance": self.solar_irradiance,
            "uv_index": self.uv_index,
            "radiation_level": self.radiation_level,
            "dust_storm": {
                "active": self.dust_storm_active,
                "intensity": self.dust_storm_intensity,
                "duration_remaining": self.dust_storm_duration if self.dust_storm_active else 0
            }
        }
    
    def simulate_period(self, sols=1, hours_per_step=1):
        """
        Simulate environment for a period of time
        
        Args:
            sols (int): Number of sols to simulate
            hours_per_step (float): Hours per simulation step
            
        Returns:
            list: History of conditions
        """
        # Reset history
        for key in self.history:
            self.history[key] = []
        
        # Calculate total hours
        total_hours = sols * 24
        steps = int(total_hours / hours_per_step)
        
        # Run simulation
        conditions_history = []
        for _ in range(steps):
            conditions = self.update(hours_per_step)
            conditions_history.append(conditions)
        
        return conditions_history
    
    def plot_history(self, save_path=None):
        """
        Plot environmental history
        
        Args:
            save_path (str): Path to save plot (optional)
            
        Returns:
            tuple: Figure and axes objects
        """
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Create time axis
        time = [self.history["sol"][i] + self.history["hour"][i]/24 for i in range(len(self.history["sol"]))]
        
        # Plot temperature
        axs[0, 0].plot(time, self.history["temperature"], 'r-')
        axs[0, 0].set_ylabel('Temperature (°C)')
        axs[0, 0].set_title('Temperature')
        axs[0, 0].grid(True)
        
        # Plot solar irradiance
        axs[0, 1].plot(time, self.history["solar_irradiance"], 'y-')
        axs[0, 1].set_ylabel('Solar Irradiance (W/m²)')
        axs[0, 1].set_title('Solar Irradiance')
        axs[0, 1].grid(True)
        
        # Plot pressure
        axs[1, 0].plot(time, self.history["pressure"], 'b-')
        axs[1, 0].set_ylabel('Pressure (Pa)')
        axs[1, 0].set_title('Atmospheric Pressure')
        axs[1, 0].grid(True)
        
        # Plot dust opacity
        axs[1, 1].plot(time, self.history["dust_opacity"], 'm-')
        axs[1, 1].set_ylabel('Dust Opacity')
        axs[1, 1].set_title('Dust Opacity')
        axs[1, 1].grid(True)
        
        # Set common x-axis label
        for ax in axs.flat:
            ax.set_xlabel('Sol')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Plot saved to {save_path}")
        
        return fig, axs

# Example usage
if __name__ == "__main__":
    # Create environment model for Jezero Crater
    env_model = MartianEnvironmentModel()
    
    # Simulate 10 sols
    env_model.simulate_period(sols=10)
    
    # Plot results
    env_model.plot_history("/home/ubuntu/martian_habitat_pathfinder/simulations/environment_simulation.png")
