# IDS706_Arushi_Wk2
Week 2 analyzing a dataset for ID706.

# Gold Price Analysis (IDS706 Project)  

![CI](https://img.shields.io/badge/build-passing-brightgreen)

---

## This project will:
- Use a dev container, Makefile, and GitHub Actions  
- Load and analyze gold prices (2015–2025) along with indexes: S&P 500 (SPX), Oil (USO), Silver (SLV), EUR/USD  
- Perform exploratory data analysis and visualization  
- Include statistical analysis (mean, median, mode, yearly summaries, quartiles)  
- Train a Machine Learning model (XGBoost) to predict gold prices  

---

## Project structure:
```text
gold-price-analysis/
├── gold_analysis.py              # Main script
├── gold_data_exploration.ipynb   # Jupyter notebook
├── gold_data_2015_25.csv         # Kaggle dataset (if included)
├── requirements.txt              # Dependencies
├── Makefile                      # Task automation
├── .devcontainer/
│   └── devcontainer.json
├── .github/
│   └── workflows/
│       └── ci.yml                # (optional) GitHub Actions config
├── .gitignore
└── README.md

#create a requirements.txt file with:

pylint
flake8
pytest
click
black
pytest-cov
pandas
numpy
matplotlib
seaborn
jupyter
scipy
kagglehub
scikit-learn
xgboost

##Create Makefile with:

install:
	pip install --upgrade pip
	pip install -r requirements.txt

run:
	python gold_analysis.py

notebook:
	jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root

test:
	pytest -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete