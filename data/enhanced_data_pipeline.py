"""
Enhanced Data Pipeline for Martian Habitat Pathfinder

This module provides an integrated data pipeline that supports both:
1. Traditional NASA data sources (MOLA, HiRISE, CRISM, MEDA, THEMIS)
2. PDF documents containing Mars habitat research and specifications

The pipeline handles data acquisition, preprocessing, and conversion to formats
suitable for both simulation and LLM training.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Add parent directory to path to import pdf_processor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.pdf_processor import PDFProcessor

class EnhancedDataPipeline:
    """
    Enhanced data pipeline that supports both NASA data sources and PDF documents
    for the Martian Habitat Pathfinder project.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the enhanced data pipeline with the target directory
        
        Args:
            data_dir (str): Directory to store and process data
        """
        self.data_dir = data_dir
        
        # Create subdirectories for different data types
        self.nasa_dir = os.path.join(data_dir, "nasa")
        self.pdf_dir = os.path.join(data_dir, "pdf")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.training_dir = os.path.join(data_dir, "training")
        self.schemas_dir = os.path.join(data_dir, "schemas")
        
        # Create all directories
        for directory in [self.nasa_dir, self.pdf_dir, self.processed_dir, 
                         self.training_dir, self.schemas_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Initialize NASA data pipeline
        self.nasa_pipeline = NASADataPipeline(self.nasa_dir)
        
        # Initialize PDF processor
        self.pdf_processor = PDFProcessor(data_dir=self.processed_dir, schemas_dir=self.schemas_dir)
        
        print(f"Enhanced Data Pipeline initialized with data directory: {data_dir}")
    
    def process_pdf_file(self, pdf_path: str, extract_training_data: bool = True) -> Dict[str, Any]:
        """
        Process a single PDF file to extract scientific data and optionally training data
        
        Args:
            pdf_path (str): Path to the PDF file
            extract_training_data (bool): Whether to extract training data for LLMs
            
        Returns:
            dict: Dictionary containing paths to extracted data
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Get filename without extension
        filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(filename)[0]
        
        # Copy PDF to pdf directory if it's not already there
        if not pdf_path.startswith(self.pdf_dir):
            import shutil
            pdf_copy_path = os.path.join(self.pdf_dir, filename)
            shutil.copy2(pdf_path, pdf_copy_path)
            pdf_path = pdf_copy_path
        
        results = {
            "pdf_path": pdf_path,
            "scientific_data": None,
            "training_data": None,
            "simulation_params": None,
            "ollama_format": None
        }
        
        try:
            # Extract scientific data
            scientific_data = self.pdf_processor.parse_scientific_data(pdf_path)
            scientific_data_path = os.path.join(self.processed_dir, f"{base_name}_scientific_data.json")
            
            with open(scientific_data_path, 'w') as f:
                json.dump(scientific_data, f, indent=2)
            
            results["scientific_data"] = scientific_data_path
            
            # Extract simulation parameters
            simulation_params = self.pdf_processor.extract_simulation_parameters(pdf_path)
            params_path = os.path.join(self.processed_dir, f"{base_name}_simulation_params.json")
            
            with open(params_path, 'w') as f:
                json.dump(simulation_params, f, indent=2)
            
            results["simulation_params"] = params_path
            
            # Extract training data if requested
            if extract_training_data:
                training_data = self.pdf_processor.convert_to_training_data(pdf_path)
                training_data_path = os.path.join(self.training_dir, f"{base_name}_training_data.json")
                
                with open(training_data_path, 'w') as f:
                    json.dump(training_data, f, indent=2)
                
                results["training_data"] = training_data_path
                
                # Convert to Ollama format
                ollama_format_path = os.path.join(self.training_dir, f"{base_name}_ollama_format.txt")
                self.pdf_processor.convert_to_ollama_format(training_data, ollama_format_path)
                
                results["ollama_format"] = ollama_format_path
            
            print(f"Successfully processed PDF: {filename}")
            return results
        
        except Exception as e:
            print(f"Error processing PDF {filename}: {e}")
            return results
    
    def process_pdf_directory(self, pdf_dir: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Process all PDF files in a directory
        
        Args:
            pdf_dir (str, optional): Directory containing PDF files. If None, uses self.pdf_dir
            
        Returns:
            dict: Dictionary of processed file results
        """
        if pdf_dir is None:
            pdf_dir = self.pdf_dir
        
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
        
        results = {
            "processed_files": []
        }
        
        # Process each PDF file
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                file_results = self.process_pdf_file(pdf_path)
                results["processed_files"].append(file_results)
        
        # Save combined results
        combined_results_path = os.path.join(self.processed_dir, "pdf_processing_results.json")
        with open(combined_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Processed {len(results['processed_files'])} PDF files")
        return results
    
    def fetch_nasa_data(self, data_types: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Fetch data from NASA sources
        
        Args:
            data_types (list, optional): List of data types to fetch. 
                                        If None, fetches all types.
                                        Options: 'mola', 'hirise', 'crism', 'meda', 'themis'
            
        Returns:
            dict: Dictionary of downloaded file paths by data source
        """
        if data_types is None:
            data_types = ['mola', 'hirise', 'crism', 'meda', 'themis']
        
        results = {}
        
        for data_type in data_types:
            if data_type == 'mola':
                results['mola'] = self.nasa_pipeline.fetch_mola_data()
            elif data_type == 'hirise':
                results['hirise'] = self.nasa_pipeline.fetch_hirise_dtm()
            elif data_type == 'crism':
                results['crism'] = self.nasa_pipeline.fetch_crism_data()
            elif data_type == 'meda':
                results['meda'] = self.nasa_pipeline.fetch_meda_data()
            elif data_type == 'themis':
                results['themis'] = self.nasa_pipeline.fetch_themis_data()
            else:
                print(f"Unknown data type: {data_type}")
        
        print(f"Fetched NASA data for types: {', '.join(data_types)}")
        return results
    
    def combine_data_sources(self, nasa_data: Optional[Dict[str, List[str]]] = None, 
                           pdf_results: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Any]:
        """
        Combine data from NASA sources and PDF documents
        
        Args:
            nasa_data (dict, optional): NASA data results from fetch_nasa_data
            pdf_results (dict, optional): PDF processing results from process_pdf_directory
            
        Returns:
            dict: Combined dataset for simulation and training
        """
        # Initialize combined data structure
        combined_data = {
            "environment": {
                "temperature": {},
                "pressure": {},
                "radiation": {},
                "dust": {}
            },
            "resources": {
                "water": {},
                "oxygen": {},
                "food": {},
                "power": {},
                "spare_parts": {}
            },
            "habitat": {},
            "simulation_params": {},
            "training_data": []
        }
        
        # Process NASA data if provided
        if nasa_data:
            # Extract MEDA environmental data
            if 'meda' in nasa_data and nasa_data['meda']:
                meda_file = next((f for f in nasa_data['meda'] if f.endswith('.csv')), None)
                if meda_file and os.path.exists(meda_file):
                    try:
                        meda_df = pd.read_csv(meda_file)
                        
                        # Extract temperature statistics
                        combined_data["environment"]["temperature"]["nasa_min"] = meda_df["temperature"].min()
                        combined_data["environment"]["temperature"]["nasa_max"] = meda_df["temperature"].max()
                        combined_data["environment"]["temperature"]["nasa_mean"] = meda_df["temperature"].mean()
                        
                        # Extract pressure statistics
                        combined_data["environment"]["pressure"]["nasa_min"] = meda_df["pressure"].min()
                        combined_data["environment"]["pressure"]["nasa_max"] = meda_df["pressure"].max()
                        combined_data["environment"]["pressure"]["nasa_mean"] = meda_df["pressure"].mean()
                        
                        # Extract dust opacity statistics
                        combined_data["environment"]["dust"]["nasa_min"] = meda_df["dust_opacity"].min()
                        combined_data["environment"]["dust"]["nasa_max"] = meda_df["dust_opacity"].max()
                        combined_data["environment"]["dust"]["nasa_mean"] = meda_df["dust_opacity"].mean()
                    except Exception as e:
                        print(f"Error processing MEDA data: {e}")
        
        # Process PDF data if provided
        if pdf_results and "processed_files" in pdf_results:
            for file_result in pdf_results["processed_files"]:
                # Process scientific data
                if file_result["scientific_data"] and os.path.exists(file_result["scientific_data"]):
                    try:
                        with open(file_result["scientific_data"], 'r') as f:
                            scientific_data = json.load(f)
                        
                        # Merge temperature data
                        if "temperature" in scientific_data:
                            for key, value in scientific_data["temperature"].items():
                                combined_data["environment"]["temperature"][f"pdf_{key}"] = value
                        
                        # Merge pressure data
                        if "pressure" in scientific_data:
                            for key, value in scientific_data["pressure"].items():
                                combined_data["environment"]["pressure"][f"pdf_{key}"] = value
                        
                        # Merge radiation data
                        if "radiation" in scientific_data:
                            for key, value in scientific_data["radiation"].items():
                                combined_data["environment"]["radiation"][f"pdf_{key}"] = value
                        
                        # Merge resource data
                        if "resources" in scientific_data:
                            for resource_type, resource_data in scientific_data["resources"].items():
                                if resource_type in combined_data["resources"]:
                                    for key, value in resource_data.items():
                                        combined_data["resources"][resource_type][f"pdf_{key}"] = value
                        
                        # Merge habitat data
                        if "habitat" in scientific_data:
                            for key, value in scientific_data["habitat"].items():
                                combined_data["habitat"][f"pdf_{key}"] = value
                    except Exception as e:
                        print(f"Error processing scientific data from PDF: {e}")
                
                # Process simulation parameters
                if file_result["simulation_params"] and os.path.exists(file_result["simulation_params"]):
                    try:
                        with open(file_result["simulation_params"], 'r') as f:
                            sim_params = json.load(f)
                        
                        # Add simulation parameters with PDF source identifier
                        pdf_name = os.path.basename(file_result["pdf_path"])
                        combined_data["simulation_params"][pdf_name] = sim_params
                    except Exception as e:
                        print(f"Error processing simulation parameters from PDF: {e}")
                
                # Process training data
                if file_result["training_data"] and os.path.exists(file_result["training_data"]):
                    try:
                        with open(file_result["training_data"], 'r') as f:
                            training_data = json.load(f)
                        
                        # Add training data to combined dataset
                        combined_data["training_data"].extend(training_data)
                    except Exception as e:
                        print(f"Error processing training data from PDF: {e}")
        
        # Save combined data
        combined_data_path = os.path.join(self.processed_dir, "combined_data.json")
        with open(combined_data_path, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"Combined data saved to {combined_data_path}")
        return combined_data
    
    def prepare_training_data_for_ollama(self) -> str:
        """
        Prepare combined training data for Ollama fine-tuning
        
        Returns:
            str: Path to the Ollama-formatted training data file
        """
        # Collect all training data
        all_training_data = []
        
        # Look for training data files
        for filename in os.listdir(self.training_dir):
            if filename.endswith('_training_data.json'):
                try:
                    with open(os.path.join(self.training_dir, filename), 'r') as f:
                        training_data = json.load(f)
                    all_training_data.extend(training_data)
                except Exception as e:
                    print(f"Error loading training data from {filename}: {e}")
        
        # Create combined Ollama format file
        ollama_format_path = os.path.join(self.training_dir, "combined_ollama_format.txt")
        self.pdf_processor.convert_to_ollama_format(all_training_data, ollama_format_path)
        
        print(f"Combined {len(all_training_data)} training examples for Ollama at {ollama_format_path}")
        return ollama_format_path
    
    def prepare_simulation_data(self) -> Dict[str, Any]:
        """
        Prepare data for simulation initialization
        
        Returns:
            dict: Simulation initialization parameters
        """
        # Start with default simulation parameters
        simulation_data = {
            "environment": {
                "temperature_range": [-120, 20],
                "pressure_range": [600, 700],
                "dust_opacity_range": [0.1, 0.9],
                "solar_irradiance_range": [0, 600]
            },
            "habitat": {
                "initial_power": 100,
                "initial_water": 500,
                "initial_oxygen": 200,
                "initial_food": 300,
                "initial_spare_parts": 50
            },
            "simulation": {
                "max_steps": 500,
                "difficulty": "normal"
            }
        }
        
        # Try to load combined data
        combined_data_path = os.path.join(self.processed_dir, "combined_data.json")
        if os.path.exists(combined_data_path):
            try:
                with open(combined_data_path, 'r') as f:
                    combined_data = json.load(f)
                
                # Update environment parameters if available
                if "environment" in combined_data:
                    if "temperature" in combined_data["environment"]:
                        temp_values = [v.get("value", 0) for v in combined_data["environment"]["temperature"].values() 
                                      if isinstance(v, dict) and "value" in v]
                        if temp_values:
                            simulation_data["environment"]["temperature_range"] = [min(temp_values), max(temp_values)]
                    
                    if "pressure" in combined_data["environment"]:
                        pressure_values = [v.get("value", 0) for v in combined_data["environment"]["pressure"].values() 
                                         if isinstance(v, dict) and "value" in v]
                        if pressure_values:
                            simulation_data["environment"]["pressure_range"] = [min(pressure_values), max(pressure_values)]
                
                # Update resource parameters if available
                if "resources" in combined_data:
                    for resource in ["water", "oxygen", "food", "power", "spare_parts"]:
                        if resource in combined_data["resources"]:
                            resource_values = [v.get("value", 0) for v in combined_data["resources"][resource].values() 
                                             if isinstance(v, dict) and "value" in v]
                            if resource_values:
                                # Use the maximum as initial value
                                simulation_data["habitat"][f"initial_{resource}"] = max(resource_values)
                
                # Check for simulation parameters from PDFs
                if "simulation_params" in combined_data and combined_data["simulation_params"]:
                    # Use the most recent PDF's parameters
                    latest_params = list(combined_data["simulation_params"].values())[-1]
                    
                    # Update simulation parameters
                    if "environment" in latest_params:
                        for key, value in latest_params["environment"].items():
                            if key in simulation_data["environment"]:
                                simulation_data["environment"][key] = value
                    
                    if "habitat" in latest_params:
                        for key, value in latest_params["habitat"].items():
                            if key in simulation_data["habitat"]:
                                simulation_data["habitat"][key] = value
                    
                    if "simulation" in latest_params:
                        for key, value in latest_params["simulation"].items():
                            if key in simulation_data["simulation"]:
                                simulation_data["simulation"][key] = value
            
            except Exception as e:
                print(f"Error preparing simulation data: {e}")
        
        # Save simulation initialization data
        sim_init_path = os.path.join(self.processed_dir, "simulation_init.json")
        with open(sim_init_path, 'w') as f:
            json.dump(simulation_data, f, indent=2)
        
        print(f"Simulation initialization data saved to {sim_init_path}")
        return simulation_data
    
    def run_full_pipeline(self, pdf_dir: Optional[str] = None, nasa_data_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run the full data pipeline
        
        Args:
            pdf_dir (str, optional): Directory containing PDF files
            nasa_data_types (list, optional): List of NASA data types to fetch
            
        Returns:
            dict: Results of the pipeline run
        """
        results = {
            "nasa_data": None,
            "pdf_results": None,
            "combined_data": None,
            "training_data": None,
            "simulation_data": None
        }
        
        # Step 1: Fetch NASA data
        try:
            results["nasa_data"] = self.fetch_nasa_data(nasa_data_types)
        except Exception as e:
            print(f"Error fetching NASA data: {e}")
        
        # Step 2: Process PDF files
        try:
            results["pdf_results"] = self.process_pdf_directory(pdf_dir)
        except Exception as e:
            print(f"Error processing PDF files: {e}")
        
        # Step 3: Combine data sources
        try:
            results["combined_data"] = self.combine_data_sources(
                results["nasa_data"], 
                results["pdf_results"]
            )
        except Exception as e:
            print(f"Error combining data sources: {e}")
        
        # Step 4: Prepare training data for Ollama
        try:
            results["training_data"] = self.prepare_training_data_for_ollama()
        except Exception as e:
            print(f"Error preparing training data: {e}")
        
        # Step 5: Prepare simulation data
        try:
            results["simulation_data"] = self.prepare_simulation_data()
        except Exception as e:
            print(f"Error preparing simulation data: {e}")
        
        print("Full data pipeline completed")
        return results


class NASADataPipeline:
    """
    Data acquisition pipeline for NASA Mars data sources:
    - MOLA (Mars Orbiter Laser Altimeter)
    - HiRISE (High Resolution Imaging Science Experiment)
    - CRISM (Compact Reconnaissance Imaging Spectrometer for Mars)
    - MEDA (Mars Environmental Dynamics Analyzer)
    - THEMIS (Thermal Emission Imaging System)
    """
    
    def __init__(self, data_dir):
        """
        Initialize the data pipeline with the target directory
        
        Args:
            data_dir (str): Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.base_pds_url = "https://pds-geosciences.wustl.edu/mro"
        self.base_pds_img_url = "https://pds-imaging.jpl.nasa.gov/data"
        
        # Create subdirectories for each data source
        self.mola_dir = os.path.join(data_dir, "mola")
        self.hirise_dir = os.path.join(data_dir, "hirise")
        self.crism_dir = os.path.join(data_dir, "crism")
        self.meda_dir = os.path.join(data_dir, "meda")
        self.themis_dir = os.path.join(data_dir, "themis")
        
        for directory in [self.mola_dir, self.hirise_dir, self.crism_dir, 
                         self.meda_dir, self.themis_dir]:
            os.makedirs(directory, exist_ok=True)
        
        print(f"NASA Data Pipeline initialized with data directory: {data_dir}")
    
    def fetch_mola_data(self, product_type="MEGDR", resolution="4"):
        """
        Fetch MOLA topography data
        
        Args:
            product_type (str): MEGDR (Mission Experiment Gridded Data Record) or 
                               IEGDR (Instrument Experiment Gridded Data Record)
            resolution (str): Resolution identifier (e.g., "4" for 4 pixels/degree)
        
        Returns:
            list: Paths to downloaded files
        """
        print(f"Fetching MOLA {product_type} data at resolution {resolution}...")
        
        # In a real implementation, this would connect to PDS and download actual data
        # For this simulation, we'll create placeholder files with metadata
        
        # Simulate downloading global topography data
        output_file = os.path.join(self.mola_dir, f"mola_{product_type.lower()}_{resolution}.csv")
        
        # Create sample metadata and placeholder data
        metadata = {
            "source": "Mars Orbiter Laser Altimeter",
            "product_type": product_type,
            "resolution": f"{resolution} pixels/degree",
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "data_type": "Topography (elevation in meters)"
        }
        
        # Save metadata
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create placeholder data file
        # In a real implementation, this would contain actual MOLA data
        with open(output_file, 'w') as f:
            f.write("longitude,latitude,elevation,quality\n")
            # Add some sample data points
            for lon in range(0, 360, 10):
                for lat in range(-90, 91, 10):
                    # Generate synthetic elevation based on position (just for simulation)
                    elev = 1000 * np.sin(np.radians(lon)) * np.cos(np.radians(lat)) - 2000
                    quality = np.random.randint(1, 5)
                    f.write(f"{lon},{lat},{elev},{quality}\n")
        
        print(f"MOLA data saved to {output_file}")
        return [output_file, output_file.replace(".csv", "_metadata.json")]
    
    def fetch_hirise_dtm(self, region="olympus_mons"):
        """
        Fetch HiRISE Digital Terrain Models (DTMs)
        
        Args:
            region (str): Target region on Mars
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"Fetching HiRISE DTM data for {region}...")
        
        # In a real implementation, this would download actual DTM data
        # For this simulation, we'll create placeholder files
        
        output_file = os.path.join(self.hirise_dir, f"hirise_dtm_{region}.csv")
        
        # Create sample metadata
        metadata = {
            "source": "High Resolution Imaging Science Experiment",
            "product_type": "Digital Terrain Model",
            "region": region,
            "resolution": "1 meter/pixel",
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "data_type": "Elevation (meters)"
        }
        
        # Save metadata
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create placeholder data file
        with open(output_file, 'w') as f:
            f.write("x,y,elevation,quality\n")
            # Add some sample data points (grid of 100x100)
            for x in range(100):
                for y in range(100):
                    # Generate synthetic elevation (just for simulation)
                    if region == "olympus_mons":
                        # Create a cone shape for Olympus Mons
                        dist = np.sqrt((x-50)**2 + (y-50)**2)
                        elev = 20000 * (1 - dist/70) if dist < 70 else 0
                    else:
                        # Random terrain for other regions
                        elev = 1000 * np.sin(x/10) * np.cos(y/10)
                    
                    quality = np.random.randint(1, 5)
                    f.write(f"{x},{y},{elev},{quality}\n")
        
        print(f"HiRISE DTM data saved to {output_file}")
        return [output_file, output_file.replace(".csv", "_metadata.json")]
    
    def fetch_crism_data(self, region="jezero_crater"):
        """
        Fetch CRISM hyperspectral data
        
        Args:
            region (str): Target region on Mars
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"Fetching CRISM hyperspectral data for {region}...")
        
        # In a real implementation, this would download actual CRISM data
        # For this simulation, we'll create placeholder files
        
        output_file = os.path.join(self.crism_dir, f"crism_{region}.csv")
        
        # Create sample metadata
        metadata = {
            "source": "Compact Reconnaissance Imaging Spectrometer for Mars",
            "product_type": "Hyperspectral Data",
            "region": region,
            "resolution": "18 meters/pixel",
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "data_type": "Spectral reflectance",
            "wavelength_range": "0.4-4.0 μm",
            "bands": 544
        }
        
        # Save metadata
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create placeholder data file with spectral bands
        # In a real implementation, this would contain actual spectral data
        with open(output_file, 'w') as f:
            # Create header with wavelengths
            wavelengths = [0.4 + 0.01*i for i in range(100)]  # Simplified to 100 bands
            header = "x,y," + ",".join([f"band_{w:.2f}" for w in wavelengths])
            f.write(header + "\n")
            
            # Add some sample data points (10x10 grid for simplicity)
            for x in range(10):
                for y in range(10):
                    # Start with coordinates
                    line = f"{x},{y}"
                    
                    # Generate synthetic spectral values
                    for w in wavelengths:
                        # Create different spectral signatures for different materials
                        if (x+y) % 3 == 0:  # Simulate clay minerals
                            val = 0.2 + 0.1 * np.sin(w*5) + 0.05 * np.random.random()
                        elif (x+y) % 3 == 1:  # Simulate iron oxides
                            val = 0.3 + 0.2 * np.sin(w*3) + 0.05 * np.random.random()
                        else:  # Simulate basaltic material
                            val = 0.15 + 0.05 * np.sin(w*10) + 0.03 * np.random.random()
                        
                        line += f",{val:.4f}"
                    
                    f.write(line + "\n")
        
        print(f"CRISM data saved to {output_file}")
        return [output_file, output_file.replace(".csv", "_metadata.json")]
    
    def fetch_meda_data(self, sol_range=(0, 10)):
        """
        Fetch MEDA environmental data from Perseverance rover
        
        Args:
            sol_range (tuple): Range of sols (Martian days) to fetch
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"Fetching MEDA data for sols {sol_range[0]}-{sol_range[1]}...")
        
        # In a real implementation, this would download actual MEDA data
        # For this simulation, we'll create placeholder files
        
        output_file = os.path.join(self.meda_dir, f"meda_sol_{sol_range[0]}-{sol_range[1]}.csv")
        
        # Create sample metadata
        metadata = {
            "source": "Mars Environmental Dynamics Analyzer (Perseverance Rover)",
            "sol_range": sol_range,
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "data_type": "Environmental measurements",
            "location": "Jezero Crater"
        }
        
        # Save metadata
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create placeholder data file
        with open(output_file, 'w') as f:
            f.write("sol,local_time,temperature,pressure,humidity,wind_speed,wind_direction,dust_opacity,uv_radiation\n")
            
            # Generate synthetic environmental data
            for sol in range(sol_range[0], sol_range[1]+1):
                for hour in range(0, 24):
                    # Temperature varies with time of day
                    temp_base = -60  # Base temperature in Celsius
                    temp_variation = 70  # Daily temperature swing
                    temp = temp_base + temp_variation * np.sin(np.pi * hour / 12) + np.random.normal(0, 3)
                    
                    # Pressure varies slightly with temperature
                    pressure = 730 + 20 * np.sin(np.pi * hour / 12) + np.random.normal(0, 5)
                    
                    # Humidity is generally very low
                    humidity = max(0, np.random.lognormal(mean=0.1, sigma=1.0))
                    
                    # Wind varies throughout the day
                    wind_speed = 5 + 10 * np.abs(np.sin(np.pi * hour / 8)) + np.random.normal(0, 2)
                    wind_direction = (hour * 15 + np.random.normal(0, 20)) % 360
                    
                    # Dust opacity increases during midday
                    dust_base = 0.5
                    dust_variation = 0.3
                    dust_opacity = dust_base + dust_variation * np.sin(np.pi * hour / 12) + np.random.normal(0, 0.1)
                    
                    # UV radiation follows the sun
                    uv_base = 0
                    uv_variation = 10
                    uv = max(0, uv_base + uv_variation * np.sin(np.pi * hour / 12) + np.random.normal(0, 0.5))
                    
                    f.write(f"{sol},{hour:02d}:00,{temp:.2f},{pressure:.2f},{humidity:.4f},{wind_speed:.2f},{wind_direction:.2f},{dust_opacity:.4f},{uv:.2f}\n")
        
        print(f"MEDA data saved to {output_file}")
        return [output_file, output_file.replace(".csv", "_metadata.json")]
    
    def fetch_themis_data(self, region="tharsis"):
        """
        Fetch THEMIS thermal inertia data
        
        Args:
            region (str): Target region on Mars
            
        Returns:
            list: Paths to downloaded files
        """
        print(f"Fetching THEMIS thermal inertia data for {region}...")
        
        # In a real implementation, this would download actual THEMIS data
        # For this simulation, we'll create placeholder files
        
        output_file = os.path.join(self.themis_dir, f"themis_{region}.csv")
        
        # Create sample metadata
        metadata = {
            "source": "Thermal Emission Imaging System",
            "product_type": "Thermal Inertia Mosaic",
            "region": region,
            "resolution": "100 meters/pixel",
            "download_date": datetime.now().strftime("%Y-%m-%d"),
            "data_type": "Thermal inertia (J m^-2 K^-1 s^-1/2)"
        }
        
        # Save metadata
        with open(output_file.replace(".csv", "_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create placeholder data file
        with open(output_file, 'w') as f:
            f.write("longitude,latitude,thermal_inertia,quality\n")
            
            # Define region bounds (simplified)
            if region == "tharsis":
                lon_range = (220, 280)
                lat_range = (-10, 30)
            elif region == "jezero_crater":
                lon_range = (77, 78)
                lat_range = (18, 19)
            else:
                lon_range = (0, 10)
                lat_range = (0, 10)
            
            # Generate synthetic thermal inertia data
            for lon in np.linspace(lon_range[0], lon_range[1], 50):
                for lat in np.linspace(lat_range[0], lat_range[1], 50):
                    # Generate synthetic thermal inertia values
                    # Different regions have different characteristic thermal inertia
                    if region == "tharsis":
                        # Lower thermal inertia for dusty volcanic plains
                        base_ti = 150
                        variation = 100
                    elif region == "jezero_crater":
                        # Higher and more varied for diverse crater floor materials
                        base_ti = 300
                        variation = 200
                    else:
                        base_ti = 200
                        variation = 150
                    
                    # Add spatial variation
                    ti = base_ti + variation * np.sin(np.radians(lon*5)) * np.cos(np.radians(lat*5))
                    # Add random noise
                    ti += np.random.normal(0, 20)
                    # Ensure positive values
                    ti = max(50, ti)
                    
                    quality = np.random.randint(1, 5)
                    f.write(f"{lon:.4f},{lat:.4f},{ti:.2f},{quality}\n")
        
        print(f"THEMIS data saved to {output_file}")
        return [output_file, output_file.replace(".csv", "_metadata.json")]
    
    def fetch_all_sample_data(self):
        """
        Fetch sample data from all sources for demonstration purposes
        
        Returns:
            dict: Dictionary of downloaded file paths by data source
        """
        results = {
            "mola": self.fetch_mola_data(),
            "hirise": self.fetch_hirise_dtm(),
            "crism": self.fetch_crism_data(),
            "meda": self.fetch_meda_data(),
            "themis": self.fetch_themis_data()
        }
        
        print("All sample data fetched successfully")
        return results


# Example usage
if __name__ == "__main__":
    # Set up data directory
    data_dir = "/home/ubuntu/martian_habitat_pathfinder/data"
    
    # Initialize enhanced data pipeline
    pipeline = EnhancedDataPipeline(data_dir)
    
    # Example PDF path (if available)
    pdf_path = "/path/to/example.pdf"
    
    # Run full pipeline
    if os.path.exists(pdf_path):
        # Process a single PDF file
        pdf_results = pipeline.process_pdf_file(pdf_path)
        print(f"PDF processing results: {pdf_results}")
    
    # Run full pipeline with NASA data only
    results = pipeline.run_full_pipeline(nasa_data_types=['meda', 'themis'])
    print(f"Pipeline results: {results}")
