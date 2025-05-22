# Feature Engineering Module

import os
import pandas as pd
import numpy as np
import json
from scipy import ndimage, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class MarsFeatureEngineering:
    """
    Feature engineering module for Mars data:
    - Computes terrain features from topographic data
    - Extracts spectral indices from hyperspectral data
    - Builds time-series features from environmental data
    - Creates derived features for resource mapping
    """
    
    def __init__(self, data_dir):
        """
        Initialize the feature engineering module
        
        Args:
            data_dir (str): Directory containing processed data
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        self.features_dir = os.path.join(data_dir, "features")
        os.makedirs(self.features_dir, exist_ok=True)
        
        print(f"Mars Feature Engineering initialized with data directory: {data_dir}")
    
    def compute_terrain_features(self, topo_file, window_size=3):
        """
        Compute terrain features from topographic data
        
        Args:
            topo_file (str): Path to processed topographic data file
            window_size (int): Window size for local calculations
            
        Returns:
            str: Path to terrain features file
        """
        print(f"Computing terrain features from {topo_file}...")
        
        # Load processed topographic data
        df = pd.read_csv(os.path.join(self.processed_dir, topo_file))
        
        # Check if this is global or local data
        is_global = "longitude" in df.columns and "latitude" in df.columns
        is_local = "x" in df.columns and "y" in df.columns
        
        if not is_global and not is_local:
            raise ValueError("Data must contain either lon/lat or x/y coordinates")
        
        # Create a copy for feature engineering
        df_features = df.copy()
        
        if is_global:
            # For global data (MOLA)
            # Convert to grid for neighborhood calculations
            lon_vals = sorted(df["longitude"].unique())
            lat_vals = sorted(df["latitude"].unique())
            
            # Check if we have a regular grid
            if len(lon_vals) * len(lat_vals) != len(df):
                print("Warning: Data is not on a regular grid, results may be approximate")
            
            # Create elevation grid
            elevation_grid = np.zeros((len(lat_vals), len(lon_vals)))
            
            for i, lat in enumerate(lat_vals):
                for j, lon in enumerate(lon_vals):
                    mask = (df["latitude"] == lat) & (df["longitude"] == lon)
                    if mask.any():
                        elevation_grid[i, j] = df.loc[mask, "elevation"].values[0]
            
            # Compute slope (degrees)
            # Use Sobel filter for gradient calculation
            dy, dx = np.gradient(elevation_grid)
            
            # Convert to degrees
            # Approximate cell size in meters (varies with latitude)
            # At equator, 1 degree is about 59.274 km on Mars
            cell_size_y = 59274 / len(lat_vals)  # meters per cell in y direction
            cell_size_x = np.zeros_like(lat_vals)
            for i, lat in enumerate(lat_vals):
                # Cell size in x direction varies with latitude
                cell_size_x[i] = 59274 * np.cos(np.radians(lat)) / len(lon_vals)
            
            # Compute slope for each cell
            slope_grid = np.zeros_like(elevation_grid)
            for i in range(len(lat_vals)):
                slope_x = dx[i, :] / cell_size_x[i]
                slope_y = dy[i, :] / cell_size_y
                slope_grid[i, :] = np.degrees(np.arctan(np.sqrt(slope_x**2 + slope_y**2)))
            
            # Compute aspect (degrees from north)
            aspect_grid = np.degrees(np.arctan2(dy, -dx))
            # Convert to 0-360 range
            aspect_grid = (aspect_grid + 360) % 360
            
            # Compute roughness (standard deviation of elevation in window)
            roughness_grid = ndimage.generic_filter(
                elevation_grid, np.std, size=window_size
            )
            
            # Compute TRI (Terrain Ruggedness Index)
            # Mean absolute difference between center cell and neighbors
            def tri_filter(values):
                center = values[len(values)//2]
                return np.mean(np.abs(values - center))
            
            tri_grid = ndimage.generic_filter(
                elevation_grid, tri_filter, size=window_size
            )
            
            # Map grid values back to dataframe
            slope_values = []
            aspect_values = []
            roughness_values = []
            tri_values = []
            
            for _, row in df.iterrows():
                lat = row["latitude"]
                lon = row["longitude"]
                
                i = lat_vals.index(lat)
                j = lon_vals.index(lon)
                
                slope_values.append(slope_grid[i, j])
                aspect_values.append(aspect_grid[i, j])
                roughness_values.append(roughness_grid[i, j])
                tri_values.append(tri_grid[i, j])
            
            # Add computed features to dataframe
            df_features["slope"] = slope_values
            df_features["aspect"] = aspect_values
            df_features["roughness"] = roughness_values
            df_features["tri"] = tri_values
            
        elif is_local:
            # For local data (HiRISE)
            # Convert to grid for neighborhood calculations
            x_vals = sorted(df["x"].unique())
            y_vals = sorted(df["y"].unique())
            
            # Check if we have a regular grid
            if len(x_vals) * len(y_vals) != len(df):
                print("Warning: Data is not on a regular grid, results may be approximate")
            
            # Create elevation grid
            elevation_grid = np.zeros((len(y_vals), len(x_vals)))
            
            for i, y in enumerate(y_vals):
                for j, x in enumerate(x_vals):
                    mask = (df["y"] == y) & (df["x"] == x)
                    if mask.any():
                        elevation_grid[i, j] = df.loc[mask, "elevation"].values[0]
            
            # Compute slope (degrees)
            # Use Sobel filter for gradient calculation
            dy, dx = np.gradient(elevation_grid)
            
            # For HiRISE, cell size is typically in meters already
            # Assuming 1 unit = 1 meter
            cell_size = 1.0
            
            # Compute slope
            slope_grid = np.degrees(np.arctan(np.sqrt((dx/cell_size)**2 + (dy/cell_size)**2)))
            
            # Compute aspect (degrees from north)
            aspect_grid = np.degrees(np.arctan2(dy, -dx))
            # Convert to 0-360 range
            aspect_grid = (aspect_grid + 360) % 360
            
            # Compute roughness (standard deviation of elevation in window)
            roughness_grid = ndimage.generic_filter(
                elevation_grid, np.std, size=window_size
            )
            
            # Compute TRI (Terrain Ruggedness Index)
            def tri_filter(values):
                center = values[len(values)//2]
                return np.mean(np.abs(values - center))
            
            tri_grid = ndimage.generic_filter(
                elevation_grid, tri_filter, size=window_size
            )
            
            # Map grid values back to dataframe
            slope_values = []
            aspect_values = []
            roughness_values = []
            tri_values = []
            
            for _, row in df.iterrows():
                x = row["x"]
                y = row["y"]
                
                i = y_vals.index(y)
                j = x_vals.index(x)
                
                slope_values.append(slope_grid[i, j])
                aspect_values.append(aspect_grid[i, j])
                roughness_values.append(roughness_grid[i, j])
                tri_values.append(tri_grid[i, j])
            
            # Add computed features to dataframe
            df_features["slope"] = slope_values
            df_features["aspect"] = aspect_values
            df_features["roughness"] = roughness_values
            df_features["tri"] = tri_values
        
        # Save feature file
        output_file = os.path.join(self.features_dir, "terrain_" + os.path.basename(topo_file))
        df_features.to_csv(output_file, index=False)
        
        # Save feature metadata
        metadata = {
            "source_file": topo_file,
            "feature_type": "terrain",
            "features_computed": ["slope", "aspect", "roughness", "tri"],
            "window_size": window_size,
            "units": {
                "slope": "degrees",
                "aspect": "degrees from north",
                "roughness": "meters",
                "tri": "meters"
            }
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Terrain features saved to {output_file}")
        return output_file
    
    def extract_spectral_indices(self, spectral_file, n_components=10):
        """
        Extract spectral indices from hyperspectral data
        
        Args:
            spectral_file (str): Path to processed spectral data file
            n_components (int): Number of principal components to extract
            
        Returns:
            str: Path to spectral features file
        """
        print(f"Extracting spectral indices from {spectral_file}...")
        
        # Load processed spectral data
        df = pd.read_csv(os.path.join(self.processed_dir, spectral_file))
        
        # Identify spectral band columns
        band_cols = [col for col in df.columns if col.startswith("band_")]
        
        if len(band_cols) == 0:
            raise ValueError("No spectral band columns found")
        
        # Create a copy for feature engineering
        df_features = df.copy()
        
        # Extract coordinates
        coord_cols = [col for col in df.columns if col not in band_cols]
        
        # Get spectral data as array
        X_spectral = df[band_cols].values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_spectral)
        
        # Apply PCA for dimensionality reduction
        n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe
        for i in range(n_components):
            df_features[f"pca_{i+1}"] = X_pca[:, i]
        
        # Calculate specific spectral indices for water-bearing minerals
        # These are simplified examples; real indices would be based on specific wavelengths
        
        # Extract wavelengths from band names
        wavelengths = []
        for band in band_cols:
            try:
                # Extract wavelength from band name (e.g., "band_1.23" -> 1.23)
                wl = float(band.split("_")[1])
                wavelengths.append(wl)
            except:
                print(f"Warning: Could not extract wavelength from {band}")
        
        # If we have enough bands, calculate some spectral indices
        if len(wavelengths) >= 3:
            # Find indices of bands closest to desired wavelengths
            # For example, bands near 1.4, 1.9, and 2.3 μm are sensitive to water/hydroxyl
            # These are simplified examples
            
            # Find band index closest to 1.4 μm (if available)
            idx_1_4 = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - 1.4))
            # Find band index closest to 1.9 μm (if available)
            idx_1_9 = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - 1.9))
            # Find band index closest to 2.3 μm (if available)
            idx_2_3 = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - 2.3))
            
            # Find band index closest to 1.0 μm (reference)
            idx_1_0 = min(range(len(wavelengths)), key=lambda i: abs(wavelengths[i] - 1.0))
            
            # Calculate water absorption features
            if max(wavelengths) >= 1.4:
                # Water Band Index 1 (1.4 μm / 1.0 μm)
                df_features["wbi_1"] = df[band_cols[idx_1_4]] / df[band_cols[idx_1_0]]
            
            if max(wavelengths) >= 1.9:
                # Water Band Index 2 (1.9 μm / 1.0 μm)
                df_features["wbi_2"] = df[band_cols[idx_1_9]] / df[band_cols[idx_1_0]]
            
            if max(wavelengths) >= 2.3:
                # Clay/Hydroxyl Index (2.3 μm / 1.0 μm)
                df_features["clay_idx"] = df[band_cols[idx_2_3]] / df[band_cols[idx_1_0]]
        
        # Save feature file
        output_file = os.path.join(self.features_dir, "spectral_" + os.path.basename(spectral_file))
        df_features.to_csv(output_file, index=False)
        
        # Save feature metadata
        metadata = {
            "source_file": spectral_file,
            "feature_type": "spectral",
            "pca_components": n_components,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "spectral_indices": ["wbi_1", "wbi_2", "clay_idx"] if len(wavelengths) >= 3 else []
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Spectral indices saved to {output_file}")
        return output_file
    
    def build_time_series_features(self, env_file, forecast_horizon=24):
        """
        Build time-series features from environmental data
        
        Args:
            env_file (str): Path to processed environmental data file
            forecast_horizon (int): Number of hours to forecast
            
        Returns:
            str: Path to time-series features file
        """
        print(f"Building time-series features from {env_file}...")
        
        # Load processed environmental data
        df = pd.read_csv(os.path.join(self.processed_dir, env_file))
        
        # Check if this is time-series data
        if "sol" not in df.columns or "local_time" not in df.columns:
            raise ValueError("Data must contain sol and local_time columns")
        
        # Create a copy for feature engineering
        df_features = df.copy()
        
        # Convert local_time to hour
        if "local_time" in df.columns:
            # Extract hour from time string (e.g., "12:00" -> 12)
            df_features["hour"] = df["local_time"].apply(
                lambda x: int(x.split(":")[0]) if ":" in x else int(x)
            )
        
        # Create a datetime-like index for time series analysis
        # Mars sol is about 24 hours and 40 minutes
        # For simplicity, we'll treat each sol as 24 hours
        df_features["time_idx"] = df_features["sol"] * 24 + df_features["hour"]
        
        # Sort by time index
        df_features = df_features.sort_values("time_idx")
        
        # Calculate rolling statistics for key environmental variables
        env_vars = ["temperature", "pressure", "dust_opacity", "wind_speed", "uv_radiation"]
        
        for var in env_vars:
            if var in df_features.columns:
                # 24-hour rolling average
                df_features[f"{var}_24h_avg"] = df_features[var].rolling(24, min_periods=1).mean()
                
                # 24-hour rolling standard deviation
                df_features[f"{var}_24h_std"] = df_features[var].rolling(24, min_periods=1).std()
                
                # 7-sol rolling average
                df_features[f"{var}_7sol_avg"] = df_features[var].rolling(7*24, min_periods=1).mean()
                
                # Daily min/max (per sol)
                daily_min = df_features.groupby("sol")[var].transform("min")
                daily_max = df_features.groupby("sol")[var].transform("max")
                
                df_features[f"{var}_daily_min"] = daily_min
                df_features[f"{var}_daily_max"] = daily_max
                df_features[f"{var}_daily_range"] = daily_max - daily_min
        
        # Calculate diurnal patterns
        # Fit sinusoidal model to capture daily cycles
        if "temperature" in df_features.columns:
            # Group by hour to get average temperature by time of day
            hourly_temp = df_features.groupby("hour")["temperature"].mean()
            
            # Create sinusoidal model of diurnal temperature variation
            hours = np.array(hourly_temp.index)
            temps = np.array(hourly_temp.values)
            
            # Fit sinusoid: A * sin(ω*t + φ) + C
            def sinusoid(t, A, omega, phi, C):
                return A * np.sin(omega * t + phi) + C
            
            # Initial guess
            p0 = [
                (temps.max() - temps.min()) / 2,  # Amplitude
                2 * np.pi / 24,  # Frequency (daily cycle)
                0,  # Phase
                temps.mean()  # Offset
            ]
            
            try:
                from scipy import optimize
                params, _ = optimize.curve_fit(sinusoid, hours, temps, p0=p0)
                
                # Add diurnal model parameters as features
                df_features["temp_diurnal_amplitude"] = params[0]
                df_features["temp_diurnal_phase"] = params[2]
                df_features["temp_diurnal_offset"] = params[3]
                
                # Add residual from diurnal model (anomaly)
                df_features["temp_diurnal_residual"] = df_features.apply(
                    lambda row: row["temperature"] - sinusoid(
                        row["hour"], params[0], params[1], params[2], params[3]
                    ),
                    axis=1
                )
            except:
                print("Warning: Could not fit diurnal temperature model")
        
        # Detect dust storm events
        if "dust_opacity" in df_features.columns:
            # Calculate dust opacity z-score
            dust_mean = df_features["dust_opacity"].mean()
            dust_std = df_features["dust_opacity"].std()
            
            df_features["dust_zscore"] = (df_features["dust_opacity"] - dust_mean) / dust_std
            
            # Flag potential dust storm events (z-score > 2)
            df_features["dust_storm_flag"] = (df_features["dust_zscore"] > 2).astype(int)
            
            # Calculate days since last dust storm
            storm_days = df_features[df_features["dust_storm_flag"] == 1]["sol"].unique()
            
            def days_since_storm(sol):
                if len(storm_days) == 0:
                    return -1  # No storms in dataset
                
                # Find most recent storm day
                past_storms = storm_days[storm_days <= sol]
                if len(past_storms) == 0:
                    return -1  # No past storms
                
                return sol - past_storms.max()
            
            df_features["days_since_dust_storm"] = df_features["sol"].apply(days_since_storm)
        
        # Calculate solar flux forecasts
        if "uv_radiation" in df_features.columns:
            # Simple forecast based on time of day and recent trends
            # In a real implementation, this would use more sophisticated models
            
            # Group by hour to get average UV by time of day
            hourly_uv = df_features.groupby("hour")["uv_radiation"].mean()
            
            # Create forecast columns
            for h in range(1, forecast_horizon + 1):
                forecast_col = f"solar_flux_forecast_{h}h"
                
                # For each row, forecast h hours ahead
                df_features[forecast_col] = df_features.apply(
                    lambda row: hourly_uv[(row["hour"] + h) % 24] * 
                                (1 + (row["uv_radiation"] / hourly_uv[row["hour"]] - 1) * 0.8**h),
                    axis=1
                )
        
        # Save feature file
        output_file = os.path.join(self.features_dir, "timeseries_" + os.path.basename(env_file))
        df_features.to_csv(output_file, index=False)
        
        # Save feature metadata
        metadata = {
            "source_file": env_file,
            "feature_type": "time_series",
            "rolling_statistics": [f"{var}_24h_avg" for var in env_vars if var in df_features.columns],
            "diurnal_features": ["temp_diurnal_amplitude", "temp_diurnal_phase", "temp_diurnal_offset"],
            "forecast_features": [f"solar_flux_forecast_{h}h" for h in range(1, forecast_horizon + 1)]
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Time-series features saved to {output_file}")
        return output_file
    
    def create_resource_potential_map(self, terrain_file, spectral_file=None, thermal_file=None):
        """
        Create resource potential map by combining multiple data sources
        
        Args:
            terrain_file (str): Path to terrain features file
            spectral_file (str): Path to spectral features file (optional)
            thermal_file (str): Path to thermal features file (optional)
            
        Returns:
            str: Path to resource potential map file
        """
        print("Creating resource potential map...")
        
        # Load terrain features
        terrain_df = pd.read_csv(os.path.join(self.features_dir, terrain_file))
        
        # Create a copy for resource mapping
        resource_df = terrain_df.copy()
        
        # Define coordinate columns based on data type
        if "longitude" in resource_df.columns and "latitude" in resource_df.columns:
            coord_cols = ["longitude", "latitude"]
        elif "x" in resource_df.columns and "y" in resource_df.columns:
            coord_cols = ["x", "y"]
        else:
            raise ValueError("Data must contain either lon/lat or x/y coordinates")
        
        # Add spectral features if available
        if spectral_file is not None:
            spectral_df = pd.read_csv(os.path.join(self.features_dir, spectral_file))
            
            # Check if coordinates match
            if all(col in spectral_df.columns for col in coord_cols):
                # Merge on coordinates
                resource_df = pd.merge(resource_df, spectral_df, on=coord_cols, how="left")
            else:
                print("Warning: Could not merge spectral data (coordinate mismatch)")
        
        # Add thermal features if available
        if thermal_file is not None:
            thermal_df = pd.read_csv(os.path.join(self.features_dir, thermal_file))
            
            # Check if coordinates match
            if all(col in thermal_df.columns for col in coord_cols):
                # Merge on coordinates
                resource_df = pd.merge(resource_df, thermal_df, on=coord_cols, how="left")
            else:
                print("Warning: Could not merge thermal data (coordinate mismatch)")
        
        # Calculate resource potential scores
        
        # 1. Solar potential (based on slope, aspect, elevation)
        if all(col in resource_df.columns for col in ["slope", "aspect", "elevation"]):
            # Higher elevation = better solar potential
            # South-facing slopes (aspect ~180) = better solar potential in northern hemisphere
            # North-facing slopes (aspect ~0/360) = better solar potential in southern hemisphere
            # Lower slopes = better solar potential
            
            # Normalize elevation to 0-1 range
            elev_min = resource_df["elevation"].min()
            elev_max = resource_df["elevation"].max()
            elev_norm = (resource_df["elevation"] - elev_min) / (elev_max - elev_min)
            
            # Calculate aspect score (0-1)
            # For northern hemisphere locations (positive latitude)
            # South-facing (aspect ~180) is optimal
            # For southern hemisphere (negative latitude)
            # North-facing (aspect ~0/360) is optimal
            
            def aspect_score(row):
                if "latitude" in row and row["latitude"] > 0:
                    # Northern hemisphere - south-facing is best
                    return 1 - abs(row["aspect"] - 180) / 180
                else:
                    # Southern hemisphere - north-facing is best
                    return 1 - min(row["aspect"], 360 - row["aspect"]) / 180
            
            aspect_norm = resource_df.apply(aspect_score, axis=1)
            
            # Slope score (0-1, lower slope is better)
            slope_norm = 1 - resource_df["slope"] / 90
            
            # Combined solar potential score (0-1)
            resource_df["solar_potential"] = (
                0.4 * elev_norm +  # Elevation component
                0.4 * aspect_norm +  # Aspect component
                0.2 * slope_norm  # Slope component
            )
        
        # 2. Water potential (based on spectral indices if available)
        if "wbi_1" in resource_df.columns and "wbi_2" in resource_df.columns:
            # Normalize water band indices
            wbi1_min = resource_df["wbi_1"].min()
            wbi1_max = resource_df["wbi_1"].max()
            wbi1_norm = (resource_df["wbi_1"] - wbi1_min) / (wbi1_max - wbi1_min)
            
            wbi2_min = resource_df["wbi_2"].min()
            wbi2_max = resource_df["wbi_2"].max()
            wbi2_norm = (resource_df["wbi_2"] - wbi2_min) / (wbi2_max - wbi2_min)
            
            # Combined water potential score (0-1)
            resource_df["water_potential"] = 0.5 * wbi1_norm + 0.5 * wbi2_norm
        
        # 3. Construction potential (based on slope, roughness, thermal inertia if available)
        if all(col in resource_df.columns for col in ["slope", "roughness"]):
            # Lower slope = better construction potential
            slope_norm = 1 - resource_df["slope"] / 90
            
            # Lower roughness = better construction potential
            rough_min = resource_df["roughness"].min()
            rough_max = resource_df["roughness"].max()
            rough_norm = 1 - (resource_df["roughness"] - rough_min) / (rough_max - rough_min)
            
            # Thermal inertia component (if available)
            if "thermal_inertia" in resource_df.columns:
                # Higher thermal inertia = better construction potential (more stable)
                ti_min = resource_df["thermal_inertia"].min()
                ti_max = resource_df["thermal_inertia"].max()
                ti_norm = (resource_df["thermal_inertia"] - ti_min) / (ti_max - ti_min)
                
                # Combined construction potential score (0-1)
                resource_df["construction_potential"] = (
                    0.4 * slope_norm +  # Slope component
                    0.3 * rough_norm +  # Roughness component
                    0.3 * ti_norm  # Thermal inertia component
                )
            else:
                # Without thermal inertia
                resource_df["construction_potential"] = 0.6 * slope_norm + 0.4 * rough_norm
        
        # 4. Overall habitat suitability score
        # Combine available potential scores
        potential_cols = [col for col in ["solar_potential", "water_potential", "construction_potential"] 
                         if col in resource_df.columns]
        
        if potential_cols:
            # Equal weighting for now
            resource_df["habitat_suitability"] = resource_df[potential_cols].mean(axis=1)
        
        # Save resource potential map
        output_file = os.path.join(self.features_dir, "resource_potential_map.csv")
        resource_df.to_csv(output_file, index=False)
        
        # Save metadata
        metadata = {
            "source_files": {
                "terrain": terrain_file,
                "spectral": spectral_file,
                "thermal": thermal_file
            },
            "potential_scores": potential_cols + ["habitat_suitability"],
            "coordinate_system": "areocentric" if "latitude" in resource_df.columns else "local"
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Resource potential map saved to {output_file}")
        return output_file
    
    def process_all_features(self):
        """
        Process all available data to generate features
        
        Returns:
            dict: Dictionary of feature file paths by feature type
        """
        results = {
            "terrain": [],
            "spectral": [],
            "timeseries": [],
            "resource": []
        }
        
        # Process terrain features from topographic data
        topo_files = [f for f in os.listdir(self.processed_dir) 
                     if f.startswith("processed_mola_") or f.startswith("processed_hirise_")]
        
        for file in topo_files:
            results["terrain"].append(self.compute_terrain_features(file))
        
        # Process spectral features from CRISM data
        spectral_files = [f for f in os.listdir(self.processed_dir) 
                         if f.startswith("processed_crism_")]
        
        for file in spectral_files:
            results["spectral"].append(self.extract_spectral_indices(file))
        
        # Process time-series features from MEDA data
        env_files = [f for f in os.listdir(self.processed_dir) 
                    if f.startswith("processed_meda_")]
        
        for file in env_files:
            results["timeseries"].append(self.build_time_series_features(file))
        
        # Create resource potential map
        if results["terrain"] and results["spectral"]:
            # Use first terrain and spectral files
            thermal_file = None
            if [f for f in os.listdir(self.processed_dir) if f.startswith("processed_themis_")]:
                thermal_file = "spectral_" + [f for f in os.listdir(self.processed_dir) 
                                            if f.startswith("processed_themis_")][0]
            
            results["resource"].append(
                self.create_resource_potential_map(
                    os.path.basename(results["terrain"][0]),
                    os.path.basename(results["spectral"][0]),
                    thermal_file
                )
            )
        
        print("All features processed successfully")
        return results

# Example usage
if __name__ == "__main__":
    feature_eng = MarsFeatureEngineering("/home/ubuntu/martian_habitat_pathfinder/data")
    feature_eng.process_all_features()
