# Exercise 3.21 - LazyPredict Regression (Diabetes Dataset)
import lazypredict
from lazypredict.Supervised import LazyRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=1)

reg = LazyRegressor()
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)

# Plot R-squared values
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(models.index, models['R-Squared'], '-s')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()