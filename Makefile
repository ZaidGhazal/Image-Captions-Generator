CONDA_ENV_NAME=image-caption

# Add conda to PATH
add_conda_path:
	sh ./export_path.sh

# Initialize conda
init:
	conda init bash

# Create conda environment
create_environment: init
	conda env create -f environment.yml

# Activate conda environment
activate:
	conda activate $(CONDA_ENV_NAME)

# Install pre-commit hooks
precommit:
	pre-commit install

# Run flake8 linting
lint:
	flake8 .

# Run autopep8 code formatting
format:
	autopep8 --in-place --recursive --aggressive --aggressive .

# Clean up conda environment
clean:
	conda env remove --name $(CONDA_ENV_NAME)

all: init add_conda_path create_environment activate precommit lint format
