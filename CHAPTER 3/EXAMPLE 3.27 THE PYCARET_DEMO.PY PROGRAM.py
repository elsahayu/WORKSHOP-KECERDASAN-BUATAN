# Example 3.27 The PyCaret_demo.py (disesuaikan untuk PyCaret 3.x)
import pandas as pd
from sklearn import datasets

iris = datasets.load_iris(as_frame=True)
iris.data['Target'] = iris.target
iris = iris.data
iris.head()

from pycaret.classification import setup, compare_models

setup(data=iris, target='Target')
compare_models()