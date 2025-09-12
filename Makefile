.PHONY: all clean setup-dirs install data-pipeline data-pipeline-rebuild train-pipeline streaming-inference run-all mlflow-ui stop-all help

# Default Python interpreter and virtualenv activation for Windows
PYTHON = python
VENV = .venv\Scripts\activate.bat
MLFLOW_PORT ?= 5001

# Default target
all: help

# Help target
help:
	@echo Available targets:
	@echo   make install                - Install project dependencies and set up environment
	@echo   make setup-dirs             - Create necessary directories for pipelines
	@echo   make data-pipeline          - Run the data pipeline
	@echo   make data-pipeline-rebuild  - Rebuild cached data artifacts and run data pipeline
	@echo   make train-pipeline         - Run the training pipeline
	@echo   make streaming-inference    - Run the streaming inference pipeline with the sample JSON
	@echo   make run-all                - Run all pipelines in sequence
	@echo   make mlflow-ui              - Launch the MLflow UI on localhost:$(MLFLOW_PORT)
	@echo   make stop-all               - Stop MLflow servers on port $(MLFLOW_PORT)
	@echo   make clean                  - Clean up artifacts

# Install project dependencies and set up environment
install:
	@echo Installing project dependencies and setting up environment...
	@echo Creating virtual environment...
	$(PYTHON) -m venv .venv
	@echo Activating virtual environment and installing dependencies...
	$(VENV) && $(PYTHON) -m pip install --upgrade pip
	$(VENV) && $(PYTHON) -m pip install -r requirements.txt
	@echo Installation completed successfully!
	@echo To activate the virtual environment, run: .venv\Scripts\activate.bat

# Create necessary directories (Windows)
setup-dirs:
	@echo Creating necessary directories...
	@if not exist artifacts mkdir artifacts
	@if not exist artifacts\data mkdir artifacts\data
	@if not exist artifacts\models mkdir artifacts\models
	@if not exist artifacts\encode mkdir artifacts\encode
	@if not exist artifacts\mlflow_run_artifacts mkdir artifacts\mlflow_run_artifacts
	@if not exist artifacts\mlflow_training_artifacts mkdir artifacts\mlflow_training_artifacts
	@if not exist artifacts\inference_batches mkdir artifacts\inference_batches
	@if not exist data mkdir data
	@if not exist data\processed mkdir data\processed
	@if not exist data\raw mkdir data\raw
	@echo Directories created successfully!

# Clean up (Windows)
clean:
	@echo Cleaning up artifacts...
	@if exist artifacts rmdir /s /q artifacts
	@if exist mlruns rmdir /s /q mlruns
	@echo Cleanup completed!

# Run data pipeline
data-pipeline: setup-dirs
	@echo Start running data pipeline...
	$(VENV) && $(PYTHON) pipelines\data_pipeline.py
	@echo Data pipeline completed successfully!

# Force rebuild for data pipeline
data-pipeline-rebuild: setup-dirs
	$(VENV) && $(PYTHON) -c "from pipelines.data_pipeline import data_pipeline; data_pipeline(force_rebuild=True)"

# Run training pipeline
train-pipeline: setup-dirs
	@echo Running training pipeline...
	$(VENV) && $(PYTHON) pipelines\training_pipeline.py

# Run streaming inference pipeline with sample JSON
streaming-inference: setup-dirs
	@echo Running streaming inference pipeline with sample JSON...
	$(VENV) && $(PYTHON) pipelines\streaming_inference_pipeline.py

# Run all pipelines in sequence
run-all: setup-dirs
	@echo Running all pipelines in sequence...
	@echo ========================================
	@echo Step 1: Running data pipeline
	@echo ========================================
	$(VENV) && $(PYTHON) pipelines\data_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 2: Running training pipeline
	@echo ========================================
	$(VENV) && $(PYTHON) pipelines\training_pipeline.py
	@echo.
	@echo ========================================
	@echo Step 3: Running streaming inference pipeline
	@echo ========================================
	$(VENV) && $(PYTHON) pipelines\streaming_inference_pipeline.py
	@echo.
	@echo ========================================
	@echo All pipelines completed successfully!
	@echo ========================================

# Launch MLflow UI (single worker to avoid Windows multiprocess issues)
mlflow-ui:
	@echo Launching MLflow UI...
	@echo MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)
	@echo Press Ctrl+C to stop the server
	$(VENV) && mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) --workers 1

# Stop all running MLflow servers (Windows)
stop-all:
	@echo Stopping MLflow servers on port $(MLFLOW_PORT)...
	@for /f "tokens=5" %%a in ('netstat -ano ^| findstr :$(MLFLOW_PORT)') do @taskkill /PID %%a /F >nul 2>&1
	@echo Done.
