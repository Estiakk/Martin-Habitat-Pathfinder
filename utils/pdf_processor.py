"""
PDF Ingestion and Parsing Module for Martian Habitat Pathfinder

This module provides utilities for extracting data from PDF files for use in
the Martian Habitat Pathfinder project, including:
- Extracting text and structured data from PDFs
- Converting PDFs to training data formats
- Parsing scientific papers and reports for Mars habitat data
- Validating extracted data against expected schemas
"""

import os
import re
import json
import logging
import tempfile
from typing import Dict, List, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
from PyPDF2 import PdfReader
from pdfminer.high_level import extract_text
import tabula
import pytesseract
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import jsonschema

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("pdf_processor")

class PDFProcessor:
    """
    Class for processing PDF files and extracting data for the Mars Habitat project.
    
    This class provides methods to:
    1. Extract text and structured data from PDFs
    2. Parse scientific papers for relevant Mars habitat information
    3. Convert PDFs to training data formats for LLMs
    4. Validate extracted data against expected schemas
    """
    
    def __init__(self, data_dir: str, schemas_dir: Optional[str] = None):
        """
        Initialize the PDF processor.
        
        Args:
            data_dir: Directory for storing extracted data
            schemas_dir: Directory containing JSON schemas for validation
        """
        self.data_dir = data_dir
        self.schemas_dir = schemas_dir or os.path.join(data_dir, "schemas")
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(self.schemas_dir, exist_ok=True)
        
        # Load schemas if available
        self.schemas = self._load_schemas()
        
        logger.info(f"PDF Processor initialized with data directory: {data_dir}")
    
    def _load_schemas(self) -> Dict[str, Any]:
        """Load JSON schemas for data validation."""
        schemas = {}
        
        # Define default schemas if not available on disk
        default_schemas = {
            "mars_data": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "object"},
                    "pressure": {"type": "object"},
                    "radiation": {"type": "object"},
                    "resources": {"type": "object"},
                    "habitat": {"type": "object"}
                }
            },
            "training_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "output": {"type": "string"}
                    },
                    "required": ["input", "output"]
                }
            }
        }
        
        # Try to load schemas from files
        for schema_name, default_schema in default_schemas.items():
            schema_path = os.path.join(self.schemas_dir, f"{schema_name}_schema.json")
            
            if os.path.exists(schema_path):
                try:
                    with open(schema_path, 'r') as f:
                        schemas[schema_name] = json.load(f)
                    logger.info(f"Loaded schema from {schema_path}")
                except Exception as e:
                    logger.warning(f"Failed to load schema from {schema_path}: {e}")
                    schemas[schema_name] = default_schema
            else:
                # Save default schema
                try:
                    with open(schema_path, 'w') as f:
                        json.dump(default_schema, f, indent=2)
                    schemas[schema_name] = default_schema
                    logger.info(f"Created default schema at {schema_path}")
                except Exception as e:
                    logger.warning(f"Failed to save default schema to {schema_path}: {e}")
                    schemas[schema_name] = default_schema
        
        return schemas
    
    def extract_text(self, pdf_path: str, method: str = "pdfminer") -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            method: Extraction method ('pdfminer', 'pypdf2', or 'pymupdf')
            
        Returns:
            Extracted text as a string
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            if method == "pdfminer":
                text = extract_text(pdf_path)
            elif method == "pypdf2":
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            elif method == "pymupdf":
                doc = fitz.open(pdf_path)
                text = ""
                for page in doc:
                    text += page.get_text() + "\n"
                doc.close()
            else:
                raise ValueError(f"Unsupported extraction method: {method}")
            
            logger.info(f"Successfully extracted text from {pdf_path} using {method}")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            
            # Try OCR as fallback
            try:
                logger.info(f"Attempting OCR extraction for {pdf_path}")
                return self.extract_text_with_ocr(pdf_path)
            except Exception as ocr_e:
                logger.error(f"OCR extraction failed for {pdf_path}: {ocr_e}")
                raise
    
    def extract_text_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each image
            text = ""
            for i, image in enumerate(images):
                # Save image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png') as temp:
                    image.save(temp.name)
                    # Extract text using OCR
                    page_text = pytesseract.image_to_string(temp.name)
                    text += f"Page {i+1}:\n{page_text}\n\n"
            
            logger.info(f"Successfully extracted text from {pdf_path} using OCR")
            return text
        except Exception as e:
            logger.error(f"Failed to extract text with OCR from {pdf_path}: {e}")
            raise
    
    def extract_tables(self, pdf_path: str) -> List[pd.DataFrame]:
        """
        Extract tables from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of pandas DataFrames containing extracted tables
        """
        try:
            # Extract tables using tabula
            tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            
            logger.info(f"Successfully extracted {len(tables)} tables from {pdf_path}")
            return tables
        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            return []
    
    def extract_images(self, pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        Extract images from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save extracted images
            
        Returns:
            List of paths to extracted images
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "extracted_images")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Extract images using PyMuPDF
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num, page in enumerate(doc):
                image_list = page.get_images(full=True)
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Determine image extension
                    ext = base_image["ext"]
                    
                    # Save image
                    image_filename = f"page{page_num+1}_img{img_index+1}.{ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                    
                    image_paths.append(image_path)
            
            logger.info(f"Successfully extracted {len(image_paths)} images from {pdf_path}")
            return image_paths
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {e}")
            return []
    
    def parse_scientific_data(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse scientific data related to Mars habitat from a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing extracted scientific data
        """
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Extract tables
        tables = self.extract_tables(pdf_path)
        
        # Initialize data dictionary
        data = {
            "temperature": {},
            "pressure": {},
            "radiation": {},
            "resources": {},
            "habitat": {}
        }
        
        # Parse temperature data
        temp_pattern = r"(?:temperature|temp)[^\n.]*?(-?\d+\.?\d*)\s*(?:°C|degrees?|C)"
        temp_matches = re.finditer(temp_pattern, text, re.IGNORECASE)
        for i, match in enumerate(temp_matches):
            try:
                value = float(match.group(1))
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                data["temperature"][f"value_{i+1}"] = {
                    "value": value,
                    "unit": "°C",
                    "context": context.strip()
                }
            except (ValueError, IndexError):
                continue
        
        # Parse pressure data
        pressure_pattern = r"(?:pressure|atmospheric pressure)[^\n.]*?(\d+\.?\d*)\s*(?:Pa|hPa|kPa|bar)"
        pressure_matches = re.finditer(pressure_pattern, text, re.IGNORECASE)
        for i, match in enumerate(pressure_matches):
            try:
                value = float(match.group(1))
                unit = match.group(0).split()[-1]
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                data["pressure"][f"value_{i+1}"] = {
                    "value": value,
                    "unit": unit,
                    "context": context.strip()
                }
            except (ValueError, IndexError):
                continue
        
        # Parse radiation data
        radiation_pattern = r"(?:radiation|cosmic rays?)[^\n.]*?(\d+\.?\d*)\s*(?:mSv|Sv|Gy|mGy|rad)"
        radiation_matches = re.finditer(radiation_pattern, text, re.IGNORECASE)
        for i, match in enumerate(radiation_matches):
            try:
                value = float(match.group(1))
                unit = match.group(0).split()[-1]
                context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                data["radiation"][f"value_{i+1}"] = {
                    "value": value,
                    "unit": unit,
                    "context": context.strip()
                }
            except (ValueError, IndexError):
                continue
        
        # Parse resource data
        resource_keywords = ["water", "oxygen", "food", "power", "energy"]
        for keyword in resource_keywords:
            pattern = f"(?:{keyword})[^\n.]*?(\d+\.?\d*)\s*(?:kg|liters|L|kWh|W)"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                try:
                    value = float(match.group(1))
                    unit = match.group(0).split()[-1]
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    if keyword not in data["resources"]:
                        data["resources"][keyword] = {}
                    data["resources"][keyword][f"value_{i+1}"] = {
                        "value": value,
                        "unit": unit,
                        "context": context.strip()
                    }
                except (ValueError, IndexError):
                    continue
        
        # Parse habitat data
        habitat_keywords = ["habitat", "module", "living space", "structure"]
        for keyword in habitat_keywords:
            pattern = f"(?:{keyword})[^\n.]*?(\d+\.?\d*)\s*(?:m2|m3|square meters|cubic meters)"
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for i, match in enumerate(matches):
                try:
                    value = float(match.group(1))
                    unit = match.group(0).split()[-1]
                    context = text[max(0, match.start() - 100):min(len(text), match.end() + 100)]
                    if keyword not in data["habitat"]:
                        data["habitat"][keyword] = {}
                    data["habitat"][keyword][f"value_{i+1}"] = {
                        "value": value,
                        "unit": unit,
                        "context": context.strip()
                    }
                except (ValueError, IndexError):
                    continue
        
        # Process tables for additional data
        for i, table in enumerate(tables):
            # Convert table to dictionary
            table_dict = table.to_dict()
            
            # Check if table contains relevant data
            table_str = str(table_dict).lower()
            if any(keyword in table_str for keyword in ["temperature", "pressure", "radiation", "water", "oxygen", "habitat"]):
                data[f"table_{i+1}"] = table_dict
        
        # Validate data
        if "mars_data" in self.schemas:
            try:
                jsonschema.validate(instance=data, schema=self.schemas["mars_data"])
                logger.info(f"Extracted data from {pdf_path} validated successfully")
            except jsonschema.exceptions.ValidationError as e:
                logger.warning(f"Extracted data from {pdf_path} failed validation: {e}")
        
        return data
    
    def convert_to_training_data(self, pdf_path: str, format_type: str = "qa") -> List[Dict[str, str]]:
        """
        Convert PDF content to training data for LLMs.
        
        Args:
            pdf_path: Path to the PDF file
            format_type: Type of training data format ('qa', 'completion', or 'chat')
            
        Returns:
            List of training data examples
        """
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Split text into sections
        sections = re.split(r'\n\s*\n|\r\n\s*\r\n', text)
        sections = [s.strip() for s in sections if s.strip()]
        
        training_data = []
        
        if format_type == "qa":
            # Generate question-answer pairs
            for i, section in enumerate(sections):
                if len(section) < 50:  # Skip short sections
                    continue
                
                # Generate questions based on section content
                questions = self._generate_questions(section)
                
                for question in questions:
                    training_data.append({
                        "input": question,
                        "output": section
                    })
        
        elif format_type == "completion":
            # Generate completion prompts
            for i, section in enumerate(sections):
                if len(section) < 50:  # Skip short sections
                    continue
                
                # Split section into prompt and completion
                words = section.split()
                if len(words) < 10:  # Skip short sections
                    continue
                
                split_point = len(words) // 3
                prompt = " ".join(words[:split_point])
                completion = " ".join(words[split_point:])
                
                training_data.append({
                    "input": prompt + "...",
                    "output": completion
                })
        
        elif format_type == "chat":
            # Generate chat-style training data
            for i in range(0, len(sections) - 1, 2):
                if i + 1 < len(sections):
                    training_data.append({
                        "input": sections[i],
                        "output": sections[i + 1]
                    })
        
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        # Validate training data
        if "training_data" in self.schemas:
            try:
                jsonschema.validate(instance=training_data, schema=self.schemas["training_data"])
                logger.info(f"Training data from {pdf_path} validated successfully")
            except jsonschema.exceptions.ValidationError as e:
                logger.warning(f"Training data from {pdf_path} failed validation: {e}")
        
        return training_data
    
    def _generate_questions(self, text: str) -> List[str]:
        """Generate questions based on text content."""
        questions = []
        
        # Check for Mars-related content
        if re.search(r'\b(?:Mars|Martian|Red Planet)\b', text, re.IGNORECASE):
            questions.append("What are the key facts about Mars mentioned in this document?")
            questions.append("How does this information relate to Mars habitat design?")
        
        # Check for habitat-related content
        if re.search(r'\b(?:habitat|living|module|structure)\b', text, re.IGNORECASE):
            questions.append("What habitat design considerations are mentioned in this document?")
            questions.append("What are the key features of the habitat described?")
        
        # Check for resource-related content
        if re.search(r'\b(?:water|oxygen|power|energy|food|resource)\b', text, re.IGNORECASE):
            questions.append("What resources are discussed in this document?")
            questions.append("How are resources managed according to this information?")
        
        # Check for environmental conditions
        if re.search(r'\b(?:temperature|pressure|radiation|environment|climate)\b', text, re.IGNORECASE):
            questions.append("What environmental conditions are described in this document?")
            questions.append("How do these environmental factors affect habitat design?")
        
        # If no specific questions were generated, add generic ones
        if not questions:
            questions.append("What are the main points discussed in this document?")
            questions.append("How does this information contribute to Mars habitat planning?")
        
        return questions
    
    def convert_to_ollama_format(self, training_data: List[Dict[str, str]], output_path: str) -> str:
        """
        Convert training data to Ollama-compatible format.
        
        Args:
            training_data: List of training data examples
            output_path: Path to save the formatted training data
            
        Returns:
            Path to the formatted training data file
        """
        with open(output_path, 'w') as f:
            for item in training_data:
                f.write(f"<s>[INST] {item['input']} [/INST] {item['output']}</s>\n\n")
        
        logger.info(f"Converted {len(training_data)} examples to Ollama format at {output_path}")
        return output_path
    
    def extract_simulation_parameters(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract simulation parameters from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing simulation parameters
        """
        # Extract text
        text = self.extract_text(pdf_path)
        
        # Initialize parameters dictionary
        params = {
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
        
        # Extract temperature range
        temp_range_pattern = r"temperature\s+range.*?(-?\d+\.?\d*)\s*(?:to|[-–])\s*(-?\d+\.?\d*)\s*(?:°C|degrees?|C)"
        temp_range_match = re.search(temp_range_pattern, text, re.IGNORECASE)
        if temp_range_match:
            try:
                min_temp = float(temp_range_match.group(1))
                max_temp = float(temp_range_match.group(2))
                params["environment"]["temperature_range"] = [min_temp, max_temp]
            except (ValueError, IndexError):
                pass
        
        # Extract pressure range
        pressure_range_pattern = r"pressure\s+range.*?(\d+\.?\d*)\s*(?:to|[-–])\s*(\d+\.?\d*)\s*(?:Pa|hPa|kPa|bar)"
        pressure_range_match = re.search(pressure_range_pattern, text, re.IGNORECASE)
        if pressure_range_match:
            try:
                min_pressure = float(pressure_range_match.group(1))
                max_pressure = float(pressure_range_match.group(2))
                params["environment"]["pressure_range"] = [min_pressure, max_pressure]
            except (ValueError, IndexError):
                pass
        
        # Extract resource values
        resource_patterns = {
            "power": r"(?:initial|starting)\s+power.*?(\d+\.?\d*)\s*(?:kWh|kW)",
            "water": r"(?:initial|starting)\s+water.*?(\d+\.?\d*)\s*(?:liters|L)",
            "oxygen": r"(?:initial|starting)\s+oxygen.*?(\d+\.?\d*)\s*(?:kg)",
            "food": r"(?:initial|starting)\s+food.*?(\d+\.?\d*)\s*(?:kg)",
            "spare_parts": r"(?:initial|starting)\s+spare\s+parts.*?(\d+\.?\d*)\s*(?:units)"
        }
        
        for resource, pattern in resource_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1))
                    params["habitat"][f"initial_{resource}"] = value
                except (ValueError, IndexError):
                    pass
        
        # Extract simulation parameters
        max_steps_pattern = r"(?:max|maximum)\s+steps.*?(\d+)"
        max_steps_match = re.search(max_steps_pattern, text, re.IGNORECASE)
        if max_steps_match:
            try:
                max_steps = int(max_steps_match.group(1))
                params["simulation"]["max_steps"] = max_steps
            except (ValueError, IndexError):
                pass
        
        difficulty_pattern = r"difficulty.*?(easy|normal|hard)"
        difficulty_match = re.search(difficulty_pattern, text, re.IGNORECASE)
        if difficulty_match:
            difficulty = difficulty_match.group(1).lower()
            params["simulation"]["difficulty"] = difficulty
        
        return params
    
    def process_pdf_directory(self, pdf_dir: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process all PDF files in a directory.
        
        Args:
            pdf_dir: Directory containing PDF files
            output_dir: Directory to save processed data
            
        Returns:
            Dictionary containing processed data
        """
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, "processed_pdfs")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            "scientific_data": {},
            "training_data": [],
            "simulation_parameters": {},
            "processed_files": []
        }
        
        # Process each PDF file
        for filename in os.listdir(pdf_dir):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdf_dir, filename)
                base_name = os.path.splitext(filename)[0]
                
                try:
                    # Extract scientific data
                    scientific_data = self.parse_scientific_data(pdf_path)
                    results["scientific_data"][base_name] = scientific_data
                    
                    # Save scientific data
                    scientific_data_path = os.path.join(output_dir, f"{base_name}_scientific_data.json")
                    with open(scientific_data_path, 'w') as f:
                        json.dump(scientific_data, f, indent=2)
                    
                    # Extract training data
                    training_data = self.convert_to_training_data(pdf_path)
                    results["training_data"].extend(training_data)
                    
                    # Save training data
                    training_data_path = os.path.join(output_dir, f"{base_name}_training_data.json")
                    with open(training_data_path, 'w') as f:
                        json.dump(training_data, f, indent=2)
                    
                    # Convert to Ollama format
                    ollama_format_path = os.path.join(output_dir, f"{base_name}_ollama_format.txt")
                    self.convert_to_ollama_format(training_data, ollama_format_path)
                    
                    # Extract simulation parameters
                    simulation_params = self.extract_simulation_parameters(pdf_path)
                    results["simulation_parameters"][base_name] = simulation_params
                    
                    # Save simulation parameters
                    params_path = os.path.join(output_dir, f"{base_name}_simulation_params.json")
                    with open(params_path, 'w') as f:
                        json.dump(simulation_params, f, indent=2)
                    
                    # Record processed file
                    results["processed_files"].append({
                        "filename": filename,
                        "scientific_data_path": scientific_data_path,
                        "training_data_path": training_data_path,
                        "ollama_format_path": ollama_format_path,
                        "simulation_params_path": params_path
                    })
                    
                    logger.info(f"Successfully processed {filename}")
                except Exception as e:
                    logger.error(f"Failed to process {filename}: {e}")
        
        # Save combined results
        combined_results_path = os.path.join(output_dir, "combined_results.json")
        with open(combined_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Processed {len(results['processed_files'])} PDF files")
        return results


# Example usage
if __name__ == "__main__":
    # Set up data directory
    data_dir = "/home/ubuntu/martian_habitat_pathfinder/data"
    
    # Initialize PDF processor
    processor = PDFProcessor(data_dir)
    
    # Example PDF path
    pdf_path = "/path/to/example.pdf"
    
    # Extract text
    if os.path.exists(pdf_path):
        text = processor.extract_text(pdf_path)
        print(f"Extracted {len(text)} characters of text")
        
        # Parse scientific data
        data = processor.parse_scientific_data(pdf_path)
        print(f"Extracted scientific data: {json.dumps(data, indent=2)}")
        
        # Convert to training data
        training_data = processor.convert_to_training_data(pdf_path)
        print(f"Generated {len(training_data)} training examples")
    else:
        print(f"Example PDF not found at {pdf_path}")
        print("Please provide a valid PDF path to test the functionality")
