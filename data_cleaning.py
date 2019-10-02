import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.stats.multicomp

%matplotlib inline

ehr = pd.read_csv('~/Research/PHD/project/Hua Zheng/previous code/full data - dm_longitudinal_062118.csv', delimiter=',', dtype={'study_id': 'category', 'smoke': 'category', 'PAT_STATUS_C': 'category', 'CVD': 'category'})
ehr = ehr.drop(ehr.columns[0], axis=1)
ehr