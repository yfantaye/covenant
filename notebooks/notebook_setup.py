"""
Notebook setup utilities for covenantv2 project.
Import this at the top of your notebooks for consistent setup.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Jupyter magic for autoreload
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        # Enable autoreload
        ipython.magic('autoreload 2')
        print("✅ Autoreload enabled - modules will reload automatically")
        
        # Additional useful magics
        ipython.magic('matplotlib inline')
        ipython.magic('config InlineBackend.figure_format = "retina"')
        print("✅ Matplotlib configured for inline display")
        
except ImportError:
    print("⚠️ IPython not available - autoreload not enabled")

# Import common libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 100)

# Configure matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ Covenant v2 notebook environment ready!")
print(f"📁 Project root: {project_root}")
print(f"🐍 Python: {sys.executable}")
print(f"📦 Pandas version: {pd.__version__}")
print(f"📊 Matplotlib version: {plt.matplotlib.__version__}") 