# Exercise 3.19 - Modifikasi Example 3.27 menggunakan Breast Cancer Dataset
import pandas as pd
from sklearn import datasets

cancer = datasets.load_breast_cancer(as_frame=True)
cancer.data['Target'] = cancer.target
cancer_df = cancer.data
cancer_df.head()

from pycaret.classification import setup, compare_models

setup(data=cancer_df, target='Target')
compare_models()