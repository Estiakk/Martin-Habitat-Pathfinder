# Data Preprocessing Module

import os
import pandas as pd
import numpy as np
import json
from scipy import interpolate, signal
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class MarsDataPreprocessor:
    """
    Preprocessing module for Mars data sources:
    - Handles missing values
    - Aligns coordinate systems
    - Normalizes units
    - Filters noise
    - Resamples data to consistent resolution
    """
    
    def __init__(self, data_dir):
        """
        Initialize the preprocessor with the data directory
        
        Args:
            data_dir (str): Directory containing raw data
        """
        self.data_dir = data_dir
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Define subdirectories for each data source
        self.mola_dir = os.path.join(data_dir, "mola")
        self.hirise_dir = os.path.join(data_dir, "hirise")
        self.crism_dir = os.path.join(data_dir, "crism")
        self.meda_dir = os.path.join(data_dir, "meda")
        self.themis_dir = os.path.join(data_dir, "themis")
        
        print(f"Mars Data Preprocessor initialized with data directory: {data_dir}")
    
    def handle_missing_values(self, df, method="knn", **kwargs):
        """
        Handle missing values in dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe with missing values
            method (str): Method to handle missing values ('knn', 'mean', 'median', 'interpolate')
            **kwargs: Additional parameters for the chosen method
            
        Returns:
            pd.DataFrame: Dataframe with missing values handled
        """
        print(f"Handling missing values using {method} method...")
        
        if df.isna().sum().sum() == 0:
            print("No missing values found")
            return df
        
        df_clean = df.copy()
        
        if method == "knn":
            # KNN imputation
            n_neighbors = kwargs.get("n_neighbors", 5)
            imputer = KNNImputer(n_neighbors=n_neighbors)
            
            # Store column names and index
            columns = df.columns
            index = df.index
            
            # Apply imputation
            df_clean = pd.DataFrame(
                imputer.fit_transform(df),
                columns=columns,
                index=index
            )
            
        elif method == "mean":
            # Mean imputation
            df_clean = df.fillna(df.mean())
            
        elif method == "median":
            # Median imputation
            df_clean = df.fillna(df.median())
            
        elif method == "interpolate":
            # Interpolation
            method = kwargs.get("interp_method", "linear")
            df_clean = df.interpolate(method=method)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Check if any missing values remain
        remaining_na = df_clean.isna().sum().sum()
        if remaining_na > 0:
            print(f"Warning: {remaining_na} missing values remain after {method} imputation")
            # Fill any remaining NAs with column medians as a fallback
            df_clean = df_clean.fillna(df_clean.median())
        
        print(f"Missing values handled: {df.isna().sum().sum()} -> {df_clean.isna().sum().sum()}")
        return df_clean
    
    def align_coordinates(self, df, source_type, target_crs="areocentric"):
        """
        Align coordinate systems to a common Mars reference system
        
        Args:
            df (pd.DataFrame): Input dataframe
            source_type (str): Source data type ('mola', 'hirise', 'crism', 'themis')
            target_crs (str): Target coordinate reference system
            
        Returns:
            pd.DataFrame: Dataframe with aligned coordinates
        """
        print(f"Aligning coordinates from {source_type} to {target_crs} system...")
        
        df_aligned = df.copy()
        
        # Different data sources use different coordinate systems
        if source_type == "mola":
            # MOLA typically uses areographic coordinates (planetocentric)
            if "longitude" in df.columns and "latitude" in df.columns:
                if target_crs == "areocentric":
                    # Convert areographic to areocentric latitude
                    # Formula: tan(areocentric_lat) = 0.9950 * tan(areographic_lat)
                    # Mars has a flattening of about 0.5%
                    df_aligned["latitude"] = np.degrees(
                        np.arctan(0.9950 * np.tan(np.radians(df["latitude"])))
                    )
                    print("Converted MOLA latitudes from areographic to areocentric")
        
        elif source_type == "hirise":
            # HiRISE DTMs often use local cartesian coordinates
            if "x" in df.columns and "y" in df.columns:
                # This would require metadata about the DTM's location on Mars
                # For simulation, we'll just note that this would happen
                print("Note: Real implementation would convert HiRISE local coordinates to global")
                
        elif source_type == "crism":
            # CRISM data might use a different projection
            # For simulation, we'll just note that this would happen
            print("Note: Real implementation would reproject CRISM data to standard coordinates")
            
        elif source_type == "themis":
            # THEMIS typically uses areographic coordinates
            if "longitude" in df.columns and "latitude" in df.columns:
                if target_crs == "areocentric":
                    # Convert areographic to areocentric latitude
                    df_aligned["latitude"] = np.degrees(
                        np.arctan(0.9950 * np.tan(np.radians(df["latitude"])))
                    )
                    print("Converted THEMIS latitudes from areographic to areocentric")
        
        # Add coordinate system metadata
        df_aligned.attrs["coordinate_system"] = target_crs
        
        return df_aligned
    
    def normalize_units(self, df, source_type):
        """
        Normalize units to standard values
        
        Args:
            df (pd.DataFrame): Input dataframe
            source_type (str): Source data type
            
        Returns:
            pd.DataFrame: Dataframe with normalized units
        """
        print(f"Normalizing units for {source_type} data...")
        
        df_norm = df.copy()
        
        if source_type == "mola":
            # MOLA elevation is typically in meters, which is our standard
            if "elevation" in df.columns:
                # No conversion needed, but we'll add metadata
                df_norm.attrs["elevation_unit"] = "meters"
                
        elif source_type == "hirise":
            # HiRISE DTMs are typically in meters, which is our standard
            if "elevation" in df.columns:
                # No conversion needed, but we'll add metadata
                df_norm.attrs["elevation_unit"] = "meters"
                
        elif source_type == "meda":
            # Convert temperature from Celsius to Kelvin
            if "temperature" in df.columns:
                df_norm["temperature"] = df["temperature"] + 273.15
                df_norm.attrs["temperature_unit"] = "kelvin"
                print("Converted temperature from Celsius to Kelvin")
                
            # Convert pressure to standard Pascal
            if "pressure" in df.columns:
                # MEDA reports pressure in Pascal, which is our standard
                df_norm.attrs["pressure_unit"] = "pascal"
                
            # Wind speed to m/s
            if "wind_speed" in df.columns:
                # MEDA reports wind speed in m/s, which is our standard
                df_norm.attrs["wind_speed_unit"] = "m/s"
                
        elif source_type == "themis":
            # Thermal inertia units are typically J m^-2 K^-1 s^-1/2
            if "thermal_inertia" in df.columns:
                # No conversion needed, but we'll add metadata
                df_norm.attrs["thermal_inertia_unit"] = "J m^-2 K^-1 s^-1/2"
        
        return df_norm
    
    def filter_noise(self, df, columns=None, method="savgol", **kwargs):
        """
        Filter noise from data
        
        Args:
            df (pd.DataFrame): Input dataframe
            columns (list): Columns to filter (None for all numeric columns)
            method (str): Filtering method ('savgol', 'median', 'gaussian')
            **kwargs: Additional parameters for the chosen method
            
        Returns:
            pd.DataFrame: Filtered dataframe
        """
        print(f"Filtering noise using {method} method...")
        
        df_filtered = df.copy()
        
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = df.select_dtypes(include=np.number).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Column {col} not found in dataframe")
                continue
                
            if method == "savgol":
                # Savitzky-Golay filter
                window_length = kwargs.get("window_length", 5)
                polyorder = kwargs.get("polyorder", 2)
                
                # Ensure window_length is odd
                if window_length % 2 == 0:
                    window_length += 1
                
                # Ensure window_length is less than data length
                window_length = min(window_length, len(df[col]) - 1)
                
                # Ensure polyorder is less than window_length
                polyorder = min(polyorder, window_length - 1)
                
                if len(df[col]) > window_length:
                    df_filtered[col] = signal.savgol_filter(
                        df[col], window_length, polyorder
                    )
                    
            elif method == "median":
                # Median filter
                kernel_size = kwargs.get("kernel_size", 3)
                
                # Ensure kernel_size is odd
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                # Ensure kernel_size is less than data length
                kernel_size = min(kernel_size, len(df[col]) - 1)
                
                if len(df[col]) > kernel_size:
                    df_filtered[col] = signal.medfilt(
                        df[col], kernel_size=kernel_size
                    )
                    
            elif method == "gaussian":
                # Gaussian filter
                sigma = kwargs.get("sigma", 1.0)
                df_filtered[col] = signal.gaussian_filter1d(
                    df[col], sigma=sigma
                )
                
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return df_filtered
    
    def resample_data(self, df, source_type, target_resolution):
        """
        Resample data to a target resolution
        
        Args:
            df (pd.DataFrame): Input dataframe
            source_type (str): Source data type
            target_resolution (float): Target resolution in degrees/pixel or meters/pixel
            
        Returns:
            pd.DataFrame: Resampled dataframe
        """
        print(f"Resampling {source_type} data to {target_resolution} resolution...")
        
        # Different resampling approaches based on data type
        if source_type in ["mola", "themis"]:
            # Global datasets with lat/lon coordinates
            if "longitude" in df.columns and "latitude" in df.columns:
                # Create a regular grid at the target resolution
                lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
                lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
                
                # Create new coordinate grid
                new_lons = np.arange(lon_min, lon_max, target_resolution)
                new_lats = np.arange(lat_min, lat_max, target_resolution)
                
                # Create meshgrid for interpolation
                lon_grid, lat_grid = np.meshgrid(new_lons, new_lats)
                
                # Identify data columns to resample (exclude coordinates)
                data_cols = [col for col in df.columns if col not in ["longitude", "latitude"]]
                
                # Initialize new dataframe
                resampled_data = []
                
                for lon in new_lons:
                    for lat in new_lats:
                        row = {"longitude": lon, "latitude": lat}
                        
                        # Find nearest neighbors for this point
                        # In a real implementation, this would use proper interpolation
                        mask = (
                            (df["longitude"] >= lon - target_resolution) &
                            (df["longitude"] < lon + target_resolution) &
                            (df["latitude"] >= lat - target_resolution) &
                            (df["latitude"] < lat + target_resolution)
                        )
                        
                        neighbors = df[mask]
                        
                        if len(neighbors) > 0:
                            # Average values from neighbors
                            for col in data_cols:
                                row[col] = neighbors[col].mean()
                        else:
                            # No neighbors found, use NaN
                            for col in data_cols:
                                row[col] = np.nan
                        
                        resampled_data.append(row)
                
                # Create new dataframe
                df_resampled = pd.DataFrame(resampled_data)
                
                # Handle any missing values from resampling
                df_resampled = self.handle_missing_values(df_resampled, method="knn")
                
                return df_resampled
                
        elif source_type in ["hirise", "crism"]:
            # Local datasets with x/y coordinates
            if "x" in df.columns and "y" in df.columns:
                # Similar approach as above but for x/y coordinates
                # For simulation, we'll just return the original data
                print("Note: Real implementation would resample local datasets to target resolution")
                return df
                
        elif source_type == "meda":
            # Time series data
            if "sol" in df.columns and "local_time" in df.columns:
                # For time series, resampling means changing the time interval
                # For simulation, we'll just return the original data
                print("Note: Real implementation would resample time series to target time interval")
                return df
        
        # Default: return original data
        print("Warning: Resampling not implemented for this data type, returning original data")
        return df
    
    def preprocess_mola_data(self, filename, output_prefix="processed_"):
        """
        Preprocess MOLA topography data
        
        Args:
            filename (str): Input filename
            output_prefix (str): Prefix for output filename
            
        Returns:
            str: Path to preprocessed file
        """
        print(f"Preprocessing MOLA data: {filename}")
        
        # Load data
        df = pd.read_csv(os.path.join(self.mola_dir, filename))
        
        # Handle missing values
        df = self.handle_missing_values(df, method="knn")
        
        # Align coordinates
        df = self.align_coordinates(df, "mola", target_crs="areocentric")
        
        # Normalize units
        df = self.normalize_units(df, "mola")
        
        # Filter noise
        df = self.filter_noise(df, columns=["elevation"], method="savgol", 
                              window_length=5, polyorder=2)
        
        # Resample data (optional)
        # df = self.resample_data(df, "mola", target_resolution=1.0)
        
        # Save preprocessed data
        output_file = os.path.join(self.processed_dir, output_prefix + filename)
        df.to_csv(output_file, index=False)
        
        # Save processing metadata
        metadata = {
            "source_file": filename,
            "preprocessing_steps": [
                "missing_value_imputation",
                "coordinate_alignment",
                "unit_normalization",
                "noise_filtering"
            ],
            "coordinate_system": "areocentric"
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessed MOLA data saved to {output_file}")
        return output_file
    
    def preprocess_hirise_data(self, filename, output_prefix="processed_"):
        """
        Preprocess HiRISE DTM data
        
        Args:
            filename (str): Input filename
            output_prefix (str): Prefix for output filename
            
        Returns:
            str: Path to preprocessed file
        """
        print(f"Preprocessing HiRISE data: {filename}")
        
        # Load data
        df = pd.read_csv(os.path.join(self.hirise_dir, filename))
        
        # Handle missing values
        df = self.handle_missing_values(df, method="knn")
        
        # Normalize units
        df = self.normalize_units(df, "hirise")
        
        # Filter noise
        df = self.filter_noise(df, columns=["elevation"], method="median", 
                              kernel_size=3)
        
        # Save preprocessed data
        output_file = os.path.join(self.processed_dir, output_prefix + filename)
        df.to_csv(output_file, index=False)
        
        # Save processing metadata
        metadata = {
            "source_file": filename,
            "preprocessing_steps": [
                "missing_value_imputation",
                "unit_normalization",
                "noise_filtering"
            ]
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessed HiRISE data saved to {output_file}")
        return output_file
    
    def preprocess_crism_data(self, filename, output_prefix="processed_"):
        """
        Preprocess CRISM hyperspectral data
        
        Args:
            filename (str): Input filename
            output_prefix (str): Prefix for output filename
            
        Returns:
            str: Path to preprocessed file
        """
        print(f"Preprocessing CRISM data: {filename}")
        
        # Load data
        df = pd.read_csv(os.path.join(self.crism_dir, filename))
        
        # Handle missing values
        df = self.handle_missing_values(df, method="knn")
        
        # Filter noise (for spectral bands)
        # Get spectral band columns
        band_cols = [col for col in df.columns if col.startswith("band_")]
        df = self.filter_noise(df, columns=band_cols, method="savgol", 
                              window_length=5, polyorder=2)
        
        # Save preprocessed data
        output_file = os.path.join(self.processed_dir, output_prefix + filename)
        df.to_csv(output_file, index=False)
        
        # Save processing metadata
        metadata = {
            "source_file": filename,
            "preprocessing_steps": [
                "missing_value_imputation",
                "spectral_noise_filtering"
            ]
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessed CRISM data saved to {output_file}")
        return output_file
    
    def preprocess_meda_data(self, filename, output_prefix="processed_"):
        """
        Preprocess MEDA environmental data
        
        Args:
            filename (str): Input filename
            output_prefix (str): Prefix for output filename
            
        Returns:
            str: Path to preprocessed file
        """
        print(f"Preprocessing MEDA data: {filename}")
        
        # Load data
        df = pd.read_csv(os.path.join(self.meda_dir, filename))
        
        # Handle missing values
        df = self.handle_missing_values(df, method="interpolate", interp_method="cubic")
        
        # Normalize units
        df = self.normalize_units(df, "meda")
        
        # Filter noise
        env_cols = ["temperature", "pressure", "humidity", "wind_speed", 
                   "dust_opacity", "uv_radiation"]
        df = self.filter_noise(df, columns=env_cols, method="savgol", 
                              window_length=7, polyorder=3)
        
        # Save preprocessed data
        output_file = os.path.join(self.processed_dir, output_prefix + filename)
        df.to_csv(output_file, index=False)
        
        # Save processing metadata
        metadata = {
            "source_file": filename,
            "preprocessing_steps": [
                "missing_value_imputation",
                "unit_normalization",
                "noise_filtering"
            ],
            "temperature_unit": "kelvin",
            "pressure_unit": "pascal",
            "wind_speed_unit": "m/s"
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessed MEDA data saved to {output_file}")
        return output_file
    
    def preprocess_themis_data(self, filename, output_prefix="processed_"):
        """
        Preprocess THEMIS thermal inertia data
        
        Args:
            filename (str): Input filename
            output_prefix (str): Prefix for output filename
            
        Returns:
            str: Path to preprocessed file
        """
        print(f"Preprocessing THEMIS data: {filename}")
        
        # Load data
        df = pd.read_csv(os.path.join(self.themis_dir, filename))
        
        # Handle missing values
        df = self.handle_missing_values(df, method="knn")
        
        # Align coordinates
        df = self.align_coordinates(df, "themis", target_crs="areocentric")
        
        # Normalize units
        df = self.normalize_units(df, "themis")
        
        # Filter noise
        df = self.filter_noise(df, columns=["thermal_inertia"], method="median", 
                              kernel_size=3)
        
        # Save preprocessed data
        output_file = os.path.join(self.processed_dir, output_prefix + filename)
        df.to_csv(output_file, index=False)
        
        # Save processing metadata
        metadata = {
            "source_file": filename,
            "preprocessing_steps": [
                "missing_value_imputation",
                "coordinate_alignment",
                "unit_normalization",
                "noise_filtering"
            ],
            "coordinate_system": "areocentric",
            "thermal_inertia_unit": "J m^-2 K^-1 s^-1/2"
        }
        
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Preprocessed THEMIS data saved to {output_file}")
        return output_file
    
    def preprocess_all_data(self):
        """
        Preprocess all available data
        
        Returns:
            dict: Dictionary of preprocessed file paths by data source
        """
        results = {
            "mola": [],
            "hirise": [],
            "crism": [],
            "meda": [],
            "themis": []
        }
        
        # Process MOLA data
        for file in os.listdir(self.mola_dir):
            if file.endswith(".csv") and not file.endswith("_metadata.csv"):
                results["mola"].append(self.preprocess_mola_data(file))
        
        # Process HiRISE data
        for file in os.listdir(self.hirise_dir):
            if file.endswith(".csv") and not file.endswith("_metadata.csv"):
                results["hirise"].append(self.preprocess_hirise_data(file))
        
        # Process CRISM data
        for file in os.listdir(self.crism_dir):
            if file.endswith(".csv") and not file.endswith("_metadata.csv"):
                results["crism"].append(self.preprocess_crism_data(file))
        
        # Process MEDA data
        for file in os.listdir(self.meda_dir):
            if file.endswith(".csv") and not file.endswith("_metadata.csv"):
                results["meda"].append(self.preprocess_meda_data(file))
        
        # Process THEMIS data
        for file in os.listdir(self.themis_dir):
            if file.endswith(".csv") and not file.endswith("_metadata.csv"):
                results["themis"].append(self.preprocess_themis_data(file))
        
        print("All data preprocessed successfully")
        return results

# Example usage
if __name__ == "__main__":
    preprocessor = MarsDataPreprocessor("/home/ubuntu/martian_habitat_pathfinder/data")
    preprocessor.preprocess_all_data()
