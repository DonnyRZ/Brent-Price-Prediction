"""
Usage in notebooks:
    from lib_import import *
"""

# Core Data Science Libraries
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn - Preprocessing
from sklearn.preprocessing import StandardScaler

# Scikit-learn - Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Utilities
import warnings

# Configuration
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print('lib_import.py imported successfully')
