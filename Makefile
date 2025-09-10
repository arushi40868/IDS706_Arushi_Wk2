# Makefile for Gold Price Analysis Project

# Install dependencies
install:
	pip install --upgrade pip
	pip install -r requirements.txt

# Run your main analysis script
run:
	python gold_analysis.py

# Launch Jupyter Lab
notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Run tests (if you add a tests/ folder later)
test:
	pytest -v

# Clean up Python cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
