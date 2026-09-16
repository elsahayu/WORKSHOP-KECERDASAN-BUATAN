from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X, y = load_diabetes(return_X_y=True)
y = (y > y.mean()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Total points: %d Correctly labeled points : %d" % (y_test.shape[0], (y_test == y_pred).sum()))