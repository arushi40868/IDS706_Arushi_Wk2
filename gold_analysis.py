import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mdanwarhossain200110/gold-price-2015-2025")

print("Path to dataset files:", path)
