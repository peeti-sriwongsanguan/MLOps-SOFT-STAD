.PHONY: setup run train test docker-build docker-run mlflow clean

# Determine the shell
SHELL := /bin/bash

# Check if conda is available
CONDA_EXE := $(shell command -v conda 2> /dev/null)

setup:
ifdef CONDA_EXE
	@echo "Conda found, setting up environment..."
	conda env create -f environment.yml
	@echo "To activate the environment, run:"
	@echo "conda init $${SHELL##*/}"
	@echo "Then close and reopen your terminal, or run:"
	@echo "source $$(conda info --base)/etc/profile.d/conda.sh"
	@echo "Finally, activate the environment with:"
	@echo "conda activate walmart-time-series"
else
	@echo "Conda not found. Please install Conda and try again."
	@echo "You can download Conda from: https://docs.conda.io/en/latest/miniconda.html"
	exit 1
endif

run:
ifdef CONDA_EXE
	conda run -n walmart-time-series python src/train.py
else
	@echo "Conda not found. Please run 'make setup' first."
	exit 1
endif

train: run

test:
ifdef CONDA_EXE
	conda run -n walmart-time-series pytest tests/
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
	conda run -n walmart-time-series mlflow ui --backend-store-uri mlflow/mlruns
else
	@echo "Conda not found. Please run 'make setup' first."
	exit 1
endif

clean:
ifdef CONDA_EXE
	conda env remove -n walmart-time-series
endif
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf mlflow/mlruns