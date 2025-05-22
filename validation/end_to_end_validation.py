"""
Validation script for end-to-end workflows in the Martian Habitat Pathfinder system.

This script tests all major workflows to ensure they function correctly:
1. PDF upload and processing
2. Data pipeline execution
3. Ollama integration
4. Simulation execution
5. UI functionality

Run this script to validate the entire system before deployment.
"""

import os
import sys
import json
import time
import logging
import requests
import unittest
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MartianHabitatPathfinderValidation(unittest.TestCase):
    """Test suite for validating end-to-end workflows."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Define paths
        cls.data_dir = project_root / "data"
        cls.upload_dir = cls.data_dir / "uploads"
        cls.processed_dir = cls.data_dir / "processed"
        cls.training_dir = cls.data_dir / "training"
        
        # Ensure directories exist
        cls.upload_dir.mkdir(exist_ok=True, parents=True)
        cls.processed_dir.mkdir(exist_ok=True, parents=True)
        cls.training_dir.mkdir(exist_ok=True, parents=True)
        
        # Create sample PDF if needed
        cls.sample_pdf_path = cls.upload_dir / "sample_mars_habitat.pdf"
        if not cls.sample_pdf_path.exists():
            logger.info("Creating sample PDF for testing")
            cls._create_sample_pdf()
        
        # Start Flask app in a separate process
        cls._start_flask_app()
        
        # Wait for app to start
        time.sleep(2)
        
        # Base URL for API requests
        cls.base_url = "http://localhost:5000"
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Stop Flask app
        cls._stop_flask_app()
    
    @classmethod
    def _create_sample_pdf(cls):
        """Create a sample PDF for testing."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            c = canvas.Canvas(str(cls.sample_pdf_path), pagesize=letter)
            c.setFont("Helvetica", 12)
            
            # Add title
            c.drawString(100, 750, "Mars Habitat Research Paper")
            c.drawString(100, 730, "Author: Dr. Mars Researcher")
            
            # Add content
            y = 700
            for i, line in enumerate([
                "Abstract:",
                "This paper presents research on Mars habitat design and resource management.",
                "",
                "1. Introduction:",
                "Mars habitats require careful resource management to ensure crew survival.",
                "Key resources include power, water, oxygen, and food.",
                "",
                "2. Resource Requirements:",
                "Power: 25 kW average consumption",
                "Water: 4 liters per person per day",
                "Oxygen: 0.84 kg per person per day",
                "Food: 1.8 kg per person per day",
                "",
                "3. Environmental Conditions:",
                "Temperature: -60°C average",
                "Pressure: 600 Pa",
                "Radiation: 0.5 mSv/day",
                "Dust storms: 20% probability annually",
                "",
                "4. Resource Management Strategies:",
                "ISRU for water extraction: 40% efficiency",
                "Solar panel efficiency: 22%",
                "Oxygen regeneration: 95% recycling rate",
                "Food production: 30% of requirements",
                "",
                "5. Conclusion:",
                "Effective resource management is critical for Mars habitat sustainability.",
                "AI-driven approaches show promise for optimizing resource allocation."
            ]):
                if i % 30 == 0 and i > 0:
                    c.showPage()
                    c.setFont("Helvetica", 12)
                    y = 750
                c.drawString(100, y, line)
                y -= 15
            
            c.save()
            logger.info(f"Created sample PDF at {cls.sample_pdf_path}")
        except Exception as e:
            logger.error(f"Error creating sample PDF: {e}")
            # Create empty file as fallback
            with open(cls.sample_pdf_path, 'w') as f:
                f.write("Sample PDF content")
    
    @classmethod
    def _start_flask_app(cls):
        """Start Flask app in a separate process."""
        import subprocess
        import signal
        
        flask_app_path = project_root / "ui_app" / "src" / "main.py"
        
        try:
            # Start Flask app
            cls.flask_process = subprocess.Popen(
                ["python", str(flask_app_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            logger.info(f"Started Flask app (PID: {cls.flask_process.pid})")
        except Exception as e:
            logger.error(f"Error starting Flask app: {e}")
            cls.flask_process = None
    
    @classmethod
    def _stop_flask_app(cls):
        """Stop Flask app."""
        import os
        import signal
        
        if hasattr(cls, 'flask_process') and cls.flask_process:
            try:
                os.killpg(os.getpgid(cls.flask_process.pid), signal.SIGTERM)
                logger.info("Stopped Flask app")
            except Exception as e:
                logger.error(f"Error stopping Flask app: {e}")
    
    def test_01_pdf_upload_and_processing(self):
        """Test PDF upload and processing workflow."""
        logger.info("Testing PDF upload and processing workflow")
        
        # Test PDF upload
        with open(self.sample_pdf_path, 'rb') as f:
            files = {'pdf_file': f}
            data = {'auto_process': 'on'}
            response = requests.post(f"{self.base_url}/pdf/upload", files=files, data=data)
        
        self.assertEqual(response.status_code, 200, "PDF upload failed")
        
        # Check if PDF was processed
        pdf_name = self.sample_pdf_path.name
        base_name = pdf_name.split('.')[0]
        scientific_data_path = self.processed_dir / f"{base_name}_scientific_data.json"
        
        # Wait for processing to complete (max 10 seconds)
        for _ in range(10):
            if scientific_data_path.exists():
                break
            time.sleep(1)
        
        self.assertTrue(scientific_data_path.exists(), "PDF processing failed")
        
        # Verify scientific data
        with open(scientific_data_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('resources', data, "Scientific data missing resources")
        self.assertIn('environment', data, "Scientific data missing environment")
    
    def test_02_data_pipeline(self):
        """Test data pipeline execution."""
        logger.info("Testing data pipeline execution")
        
        # Run data pipeline
        response = requests.post(f"{self.base_url}/data/run_pipeline")
        self.assertEqual(response.status_code, 200, "Data pipeline execution failed")
        
        # Check if combined data was created
        combined_data_path = self.processed_dir / "combined_data.json"
        
        # Wait for processing to complete (max 10 seconds)
        for _ in range(10):
            if combined_data_path.exists():
                break
            time.sleep(1)
        
        self.assertTrue(combined_data_path.exists(), "Combined data creation failed")
        
        # Verify combined data
        with open(combined_data_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('resources', data, "Combined data missing resources")
        self.assertIn('environment', data, "Combined data missing environment")
    
    def test_03_training_data_preparation(self):
        """Test training data preparation for Ollama."""
        logger.info("Testing training data preparation")
        
        # Prepare training data
        response = requests.post(f"{self.base_url}/pdf/prepare_training")
        self.assertEqual(response.status_code, 200, "Training data preparation failed")
        
        # Check if training data was created
        training_data_path = self.training_dir / "combined_training_data.json"
        
        # Wait for processing to complete (max 10 seconds)
        for _ in range(10):
            if training_data_path.exists():
                break
            time.sleep(1)
        
        self.assertTrue(training_data_path.exists(), "Training data creation failed")
    
    def test_04_simulation_data_preparation(self):
        """Test simulation data preparation."""
        logger.info("Testing simulation data preparation")
        
        # Prepare simulation data
        response = requests.post(f"{self.base_url}/pdf/prepare_simulation")
        self.assertEqual(response.status_code, 200, "Simulation data preparation failed")
        
        # Check if simulation data was created
        simulation_data_path = self.processed_dir / "simulation_init.json"
        
        # Wait for processing to complete (max 10 seconds)
        for _ in range(10):
            if simulation_data_path.exists():
                break
            time.sleep(1)
        
        self.assertTrue(simulation_data_path.exists(), "Simulation data creation failed")
    
    def test_05_ollama_status(self):
        """Test Ollama status check."""
        logger.info("Testing Ollama status check")
        
        # Check Ollama status
        response = requests.get(f"{self.base_url}/ollama/status")
        self.assertEqual(response.status_code, 200, "Ollama status check failed")
        
        # Parse response
        data = response.json()
        
        # Status might be online or offline, but response should have a status field
        self.assertIn('status', data, "Ollama status response missing status field")
    
    def test_06_simulation_start(self):
        """Test simulation start."""
        logger.info("Testing simulation start")
        
        # Start simulation
        data = {
            'simulation_type': 'standard',
            'ai_mode': 'rl',
            'duration': '10'
        }
        response = requests.post(f"{self.base_url}/simulation/start_simulation", data=data)
        self.assertEqual(response.status_code, 200, "Simulation start failed")
        
        # Check simulation status
        response = requests.get(f"{self.base_url}/simulation/status")
        self.assertEqual(response.status_code, 200, "Simulation status check failed")
        
        # Parse response
        data = response.json()
        
        # Simulation might be active or not, but response should have a status field
        self.assertIn('status', data, "Simulation status response missing status field")
    
    def test_07_ui_pages(self):
        """Test UI pages accessibility."""
        logger.info("Testing UI pages accessibility")
        
        # Test main pages
        pages = [
            "/",
            "/dashboard/",
            "/pdf/",
            "/data/",
            "/simulation/",
            "/ollama/",
            "/dashboard/workflow",
            "/dashboard/quick_start"
        ]
        
        for page in pages:
            response = requests.get(f"{self.base_url}{page}")
            self.assertEqual(response.status_code, 200, f"Page {page} not accessible")
    
    def test_08_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        logger.info("Testing complete end-to-end workflow")
        
        # 1. Upload PDF
        with open(self.sample_pdf_path, 'rb') as f:
            files = {'pdf_file': f}
            data = {'auto_process': 'on'}
            response = requests.post(f"{self.base_url}/pdf/upload", files=files, data=data)
        
        self.assertEqual(response.status_code, 200, "PDF upload failed")
        
        # 2. Run data pipeline
        response = requests.post(f"{self.base_url}/data/run_pipeline")
        self.assertEqual(response.status_code, 200, "Data pipeline execution failed")
        
        # 3. Prepare training data
        response = requests.post(f"{self.base_url}/pdf/prepare_training")
        self.assertEqual(response.status_code, 200, "Training data preparation failed")
        
        # 4. Prepare simulation data
        response = requests.post(f"{self.base_url}/pdf/prepare_simulation")
        self.assertEqual(response.status_code, 200, "Simulation data preparation failed")
        
        # 5. Start simulation
        data = {
            'simulation_type': 'standard',
            'ai_mode': 'rl',
            'duration': '10'
        }
        response = requests.post(f"{self.base_url}/simulation/start_simulation", data=data)
        self.assertEqual(response.status_code, 200, "Simulation start failed")
        
        # 6. Check simulation status
        response = requests.get(f"{self.base_url}/simulation/status")
        self.assertEqual(response.status_code, 200, "Simulation status check failed")
        
        # Wait for simulation to complete (max 30 seconds)
        for _ in range(30):
            response = requests.get(f"{self.base_url}/simulation/status")
            data = response.json()
            if data.get('status') == 'completed':
                break
            time.sleep(1)
        
        logger.info("End-to-end workflow test completed successfully")

if __name__ == "__main__":
    unittest.main()
