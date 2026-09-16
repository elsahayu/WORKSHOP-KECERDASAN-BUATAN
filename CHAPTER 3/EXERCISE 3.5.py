from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
import pickle

X, y = load_iris(return_X_y=True)

clf = GaussianNB()
clf.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(clf, f)

with open('model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

p = loaded_model.predict([[5.0, 3.4, 1.5, 0.4]])
print(p)