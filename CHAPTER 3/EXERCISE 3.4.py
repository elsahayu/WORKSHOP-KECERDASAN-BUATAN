import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot

cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)

features_to_plot = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean smoothness'
]

df[features_to_plot].hist()
pyplot.tight_layout()
pyplot.show()