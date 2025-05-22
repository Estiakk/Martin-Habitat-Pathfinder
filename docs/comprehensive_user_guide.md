# Martian Habitat Pathfinder - Comprehensive User Guide

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Installation and Setup](#installation-and-setup)
4. [PDF Management](#pdf-management)
5. [Data Management](#data-management)
6. [Ollama LLM Integration](#ollama-llm-integration)
7. [Simulation Environment](#simulation-environment)
8. [Workflows](#workflows)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)
11. [API Reference](#api-reference)

## Introduction

The Martian Habitat Pathfinder is an AI-driven resource management system designed to optimize resource allocation and decision-making for Mars habitats. This comprehensive system combines PDF data extraction, NASA data sources, reinforcement learning, and large language models to create a powerful simulation and decision-making platform.

This guide provides detailed instructions on how to use all features of the Martian Habitat Pathfinder system, from basic operations to advanced configurations.

## System Overview

The Martian Habitat Pathfinder consists of several integrated components:

1. **PDF Management**: Upload and process scientific papers and technical documents about Mars habitats to extract valuable data.

2. **Data Management**: Fetch and process data from NASA sources, combine with extracted PDF data, and prepare for simulation and AI training.

3. **Ollama LLM Integration**: Configure and fine-tune large language models using Ollama for natural language decision-making and explanations.

4. **Simulation Environment**: Run Mars habitat simulations with different AI decision modes, scenarios, and parameters.

5. **User Interface**: A comprehensive web-based interface that provides access to all system features and visualizations.

## Installation and Setup

### System Requirements

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum, 16GB recommended
- **Disk Space**: 10GB minimum for base system, additional space for models and data
- **GPU**: Optional but recommended for faster LLM inference

### Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-organization/martian-habitat-pathfinder.git
   cd martian-habitat-pathfinder
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Ollama** (for LLM support):
   - Follow the instructions at [https://ollama.ai/download](https://ollama.ai/download)
   - Ensure Ollama is running before using LLM features

5. **Initialize the System**:
   ```bash
   python setup.py
   ```

6. **Start the UI**:
   ```bash
   cd ui_app
   python src/main.py
   ```

7. **Access the UI**:
   - Open a web browser and navigate to `http://localhost:5000`

## PDF Management

The PDF Management module allows you to upload, process, and extract data from scientific papers and technical documents about Mars habitats.

### Uploading PDFs

1. Navigate to the **PDF Management** page from the main menu.
2. Click the **Browse** button in the "Upload PDF" section.
3. Select a PDF file from your computer.
4. Check "Automatically process after upload" if you want immediate processing.
5. Click the **Upload** button.

### Processing PDFs

PDFs can be processed individually or in batch:

1. **Individual Processing**:
   - Click the **Process** button next to a PDF in the list.
   - Wait for processing to complete.
   - View results by clicking the **Results** button.

2. **Batch Processing**:
   - Click the **Process All PDFs** button in the "Batch Actions" section.
   - Wait for processing to complete.
   - View results for each PDF by clicking their respective **Results** buttons.

### Viewing Processing Results

After processing, you can view the extracted data:

1. Click the **Results** button next to a processed PDF.
2. The results page shows:
   - Scientific data extracted from the PDF
   - Simulation parameters derived from the data
   - Training data prepared for LLMs
   - Ollama format data (if available)

### Managing PDFs

You can manage your uploaded PDFs:

- **View**: Open the PDF in the browser.
- **Download**: Download the PDF to your computer.
- **Delete**: Remove the PDF from the system.

## Data Management

The Data Management module allows you to fetch data from NASA sources, combine it with PDF data, and prepare it for simulation and AI training.

### Fetching NASA Data

1. Navigate to the **Data Management** page from the main menu.
2. Select the data types you want to fetch (MOLA, HiRISE, CRISM, etc.).
3. Click the **Fetch NASA Data** button.
4. Wait for the data to be downloaded and processed.

### Viewing NASA Data

1. Click on a data type in the list to view details.
2. The data view page shows:
   - Preview of the data
   - Basic visualizations
   - Metadata information

### Running the Data Pipeline

The data pipeline combines all data sources and prepares them for simulation:

1. Click the **Run Pipeline** button in the main Data Management page.
2. Wait for the pipeline to complete.
3. View the combined data by clicking **View Combined Data**.

### Viewing Combined Data

The combined data view shows:

- Environment data (temperature, pressure, radiation, etc.)
- Resource data (water, oxygen, food, power, etc.)
- Visualizations of key metrics
- Download options for the data

## Ollama LLM Integration

The Ollama LLM Integration module allows you to configure and use large language models for decision-making and explanations.

### Checking Ollama Status

1. Navigate to the **Ollama LLM** page from the main menu.
2. The status section shows if Ollama is online and available models.
3. Click **Refresh Status** to update the information.

### Pulling Models

1. Enter a model name in the "Pull Model" section (e.g., llama2, mistral, gemma).
2. Click the **Pull Model** button.
3. Wait for the model to download (this may take some time).

### Creating Custom Models

1. Enter a new model name in the "Create Custom Model" section.
2. Select a base model from the dropdown.
3. Enter a system prompt that specializes the model for Mars habitat management.
4. Click the **Create Model** button.

### Fine-tuning Models

1. Select a model to fine-tune from the dropdown.
2. Select a training data file (created from processed PDFs).
3. Click the **Fine-tune Model** button.
4. Wait for fine-tuning to complete.

### Testing Models

1. Select a model from the dropdown in the "Test Model" section.
2. Enter a prompt related to Mars habitat management.
3. Adjust temperature and max tokens as needed.
4. Click the **Generate** button to see the model's response.

### Using Models for Decision-Making

There are three ways to use models for decision-making:

1. **Generate Text**: Get free-form text responses about Mars habitat management.
2. **Generate JSON**: Get structured JSON responses following a specific schema.
3. **Select Action**: Get specific action recommendations based on habitat state.

## Simulation Environment

The Simulation Environment module allows you to run Mars habitat simulations with different AI decision modes, scenarios, and parameters.

### Starting a Simulation

1. Navigate to the **Simulation** page from the main menu.
2. Select a simulation type:
   - Standard Mars Habitat
   - Dust Storm Scenario
   - Equipment Failure Scenario
   - Long-term Planning
   - Custom Scenario
3. Select an AI decision mode:
   - Reinforcement Learning
   - Large Language Model
   - Hybrid (RL+LLM)
   - Manual Control
4. Set the duration in Mars sols.
5. Click the **Start Simulation** button.

### Configuring Simulation Settings

Before starting a simulation, you can configure initial settings:

1. Adjust initial resource levels (power, water, oxygen, food).
2. Set crew size.
3. Click **Update Settings** to save changes.

### Creating Custom Scenarios

1. Select "Custom Scenario" as the simulation type.
2. In the modal that appears:
   - Enter a scenario name and description.
   - Set dust storm and equipment failure probabilities.
   - Adjust solar panel and ISRU efficiencies.
   - Define custom events in JSON format.
3. Click **Save Scenario** to create the custom scenario.

### Monitoring Simulation

During a simulation, you can monitor:

- Current sol and elapsed time
- Resource levels and trends
- Simulation log entries
- AI decisions and explanations

### Controlling Simulation

While a simulation is running, you can:

- **Pause**: Temporarily stop the simulation.
- **Stop**: End the simulation early.
- **Export Log**: Save the simulation log to a file.
- **Clear Log**: Clear the simulation log display.

## Workflows

The Martian Habitat Pathfinder supports several workflows for different use cases.

### Quick Start Workflow

For a quick demonstration of the system:

1. Upload a PDF document on the PDF Management page.
2. Process the PDF automatically.
3. Go to Simulation and start a standard simulation.
4. View results and resource trends.

### Complete Research Workflow

For comprehensive analysis:

1. Upload multiple PDF documents.
2. Process all PDFs.
3. Fetch NASA data.
4. Run the full data pipeline to combine all sources.
5. Prepare training data for Ollama.
6. Configure and fine-tune an LLM.
7. Prepare simulation data.
8. Run a custom simulation with the fine-tuned LLM.
9. Analyze results and export logs.

### AI Comparison Workflow

To compare different AI approaches:

1. Complete the data processing workflow.
2. Run a simulation with "Reinforcement Learning" AI mode.
3. Export the simulation log.
4. Run the same simulation with "Large Language Model" AI mode.
5. Export the simulation log.
6. Run the same simulation with "Hybrid" AI mode.
7. Compare the results and resource trends across all three approaches.

## Troubleshooting

### PDF Processing Issues

- **Problem**: PDF processing fails or extracts no data.
  - **Solution**: Ensure PDFs are not password-protected or encrypted.
  - **Solution**: Check that PDFs contain text (not just scanned images).
  - **Solution**: Try processing PDFs individually rather than in batch.

- **Problem**: Extracted data is incomplete or inaccurate.
  - **Solution**: Check the PDF content for clear numerical data.
  - **Solution**: Edit the extracted data manually in the JSON files.

### Ollama Connection Issues

- **Problem**: Ollama status shows "Offline".
  - **Solution**: Ensure Ollama is installed and running on your system.
  - **Solution**: Check that the Ollama server is accessible at http://localhost:11434.
  - **Solution**: Restart Ollama if it's not responding.

- **Problem**: Model pulling fails.
  - **Solution**: Check your internet connection.
  - **Solution**: Ensure you have sufficient disk space for model downloads.
  - **Solution**: Verify the model name is correct.

### Simulation Performance Issues

- **Problem**: Simulation runs slowly.
  - **Solution**: Reduce the simulation duration for faster results.
  - **Solution**: Use RL mode instead of LLM mode for better performance.
  - **Solution**: Close other resource-intensive applications.

- **Problem**: Simulation crashes or freezes.
  - **Solution**: Check system resource usage in the dashboard.
  - **Solution**: Restart the application.
  - **Solution**: Check the application logs for error messages.

## Advanced Features

### Custom LLM Prompts

You can create custom prompts for specialized LLM behavior:

1. Navigate to the Ollama LLM page.
2. Use the "Generate Text" feature with a model.
3. Craft prompts that include specific instructions, context, and constraints.
4. Save effective prompts for future use.

### Simulation Event Scripting

Create complex simulation scenarios with custom event scripts:

1. Define events in JSON format in the Custom Scenario modal.
2. Events can include dust storms, equipment failures, supply deliveries, etc.
3. Specify the sol, type, severity, and duration for each event.

### Data Pipeline Customization

Customize the data pipeline for specific research needs:

1. Edit the configuration files in the `data` directory.
2. Modify feature extraction parameters.
3. Adjust data fusion weights.
4. Add custom data sources.

## API Reference

The Martian Habitat Pathfinder provides a RESTful API for programmatic access to all features.

### PDF Management API

- `POST /pdf/upload`: Upload a PDF file.
- `POST /pdf/process/<filename>`: Process a specific PDF file.
- `POST /pdf/process_all`: Process all uploaded PDF files.
- `GET /pdf/view/<filename>`: View a PDF file.
- `GET /pdf/download/<filename>`: Download a PDF file.
- `POST /pdf/delete/<filename>`: Delete a PDF file.
- `GET /pdf/results/<filename>`: View processing results for a PDF file.

### Data Management API

- `POST /data/fetch_nasa`: Fetch data from NASA sources.
- `GET /data/view_nasa/<data_type>`: View NASA data for a specific type.
- `GET /data/view_combined`: View combined data from all sources.
- `GET /data/view_simulation_init`: View simulation initialization data.
- `POST /data/run_pipeline`: Run the full data pipeline.
- `GET /data/download/<file_type>`: Download data files.

### Ollama API

- `GET /ollama/status`: Check Ollama server status.
- `POST /ollama/pull_model`: Pull a model from Ollama.
- `POST /ollama/create_model`: Create a custom model.
- `POST /ollama/finetune_model`: Fine-tune a model.
- `POST /ollama/generate_text`: Generate text with a model.
- `POST /ollama/generate_json`: Generate structured JSON with a model.
- `POST /ollama/select_action`: Get action recommendations for a habitat state.

### Simulation API

- `POST /simulation/start_simulation`: Start a simulation.
- `POST /simulation/pause_simulation`: Pause a running simulation.
- `POST /simulation/stop_simulation`: Stop a running simulation.
- `GET /simulation/status`: Get simulation status.
- `POST /simulation/update_settings`: Update simulation settings.

---

This comprehensive guide covers all aspects of the Martian Habitat Pathfinder system. For additional support or to report issues, please contact the development team.
