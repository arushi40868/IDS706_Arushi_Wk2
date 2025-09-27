# IDS706_Arushi_Wk2
Week 2 analyzing a dataset for ID706.

# Gold Price Analysis (IDS706 Project)  

![Python CI](https://github.com/arushi40868/IDS706_Arushi_Wk2/actions/workflows/python-ci.yml/badge.svg)


![CI](https://img.shields.io/badge/build-passing-brightgreen)

---

## This project will:
- Use a dev container, Makefile, and GitHub Actions  
- Load and analyze gold prices (2015–2025) along with indexes: S&P 500 (SPX), Oil (USO), Silver (SLV), EUR/USD  
- Perform exploratory data analysis and visualization  
- Include statistical analysis (mean, median, mode, yearly summaries, quartiles)  
- Train a Machine Learning model (XGBoost) to predict gold prices  

---
🔎 Data Analysis Steps
1. Data Exploration

Load dataset into pandas DataFrame (gold_df)

Inspect missing values and column types

Visualize trends with Matplotlib

2. Statistical Analysis

Compute descriptive stats (mean, median, mode, std)

Group by year to calculate yearly averages

Quartile analysis of gold prices

Histograms and distributions

3. Comparative Analysis

Normalize gold and indexes to compare performance (start=100)

Count days gold outperformed SPX, USO, SLV, and EUR/USD

Identify “risk-off” days (Gold ↑, Index ↓)

4. Machine Learning (XGBoost)

Train model to predict GLD using other indexes as features

Split dataset (train/test)

Evaluate with MSE, RMSE, R²

Visualize feature importance

---

## Running Test

Unit and system tests are located in the tests/ folder. They cover:
-Data loading – verifies gold dataset loads with expected columns and datetime index.
-Data cleaning – ensures missing values are dropped and column names normalized.
-ML model prediction – checks that regression produces valid numeric predictions.

Run all tests with:

python -m unittest discover -s tests

Expected output if tests pass:

...
----------------------------------------------------------------------
Ran 3 tests in 0.01s

OK


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

## Running Dev Container:
```text

clone repo:
	git clone https://github.com/arushi40868/IDS706_Arushi_Wk2.git
	cd IDS706_Arushi_Wk2

Open the project in VS Code:
	code .

Once running, you’ll be inside the container with everything set up to run:
	make install   # install dependencies
	make test      # run tests
	make lint      # run flake8 linting
	make format    # run Black auto-formatter