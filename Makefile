.PHONY: setup run train test docker-build docker-run mlflow clean

# Determine the shell
SHELL := /bin/bash

# Check if conda is available
CONDA_EXE := $(shell command -v conda 2> /dev/null)

# Define the environment name
ENV_NAME := walmart-time-series

setup:
ifdef CONDA_EXE
	@echo "Conda found, setting up environment..."
	conda env create -f environment.yml
	@echo "Environment created. To activate, run:"
	@echo "conda activate $(ENV_NAME)"
else
	@echo "Conda not found. Please install Conda and try again."
	@echo "You can download Conda from: https://docs.conda.io/en/latest/miniconda.html"
	exit 1
endif

run:
ifdef CONDA_EXE
	@echo "Checking for $(ENV_NAME) environment..."
	@if conda env list | grep -q $(ENV_NAME); then \
		echo "Environment found. Running the training script..."; \
		conda run -n $(ENV_NAME) python src/train.py; \
	else \
		echo "Environment not found. Setting up..."; \
		make setup; \
		echo "Environment created. Running the training script..."; \
		conda run -n $(ENV_NAME) python src/train.py; \
	fi
else
	@echo "Conda not found. Please install Conda and try again."
	@echo "You can download Conda from: https://docs.conda.io/en/latest/miniconda.html"
	exit 1
endif

train: run

test:
ifdef CONDA_EXE
	@if conda env list | grep -q $(ENV_NAME); then \
		conda run -n $(ENV_NAME) pytest tests/; \
	else \
		echo "Environment '$(ENV_NAME)' not found."; \
		echo "Please run 'make setup' to create the environment."; \
	fi
else
	@echo "Conda not found. Please run 'make setup' first."
	exit 1
endif

docker-build:
	docker build -t walmart-time-series .

docker-run:
	docker run -it walmart-time-series

mlflow:
ifdef CONDA_EXE
	@if conda env list | grep -q $(ENV_NAME); then \
		conda run -n $(ENV_NAME) mlflow ui --backend-store-uri mlflow/mlruns; \
	else \
		echo "Environment '$(ENV_NAME)' not found."; \
		echo "Please run 'make setup' to create the environment."; \
	fi
else
	@echo "Conda not found. Please run 'make setup' first."
	exit 1
endif

clean:
ifdef CONDA_EXE
	@if conda env list | grep -q $(ENV_NAME); then \
		conda env remove -n $(ENV_NAME); \
	else \
		echo "Environment '$(ENV_NAME)' not found. Nothing to clean."; \
	fi
endif
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf mlflow/mlruns