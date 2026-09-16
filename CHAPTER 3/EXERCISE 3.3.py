from sklearn import svm
import pandas as pd
from matplotlib import pyplot

url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
df = pd.read_csv(url)

df = df.dropna()

pyplot.scatter(df['sepal_length'], df['sepal_width'])
pyplot.xlabel('Sepal Length')
pyplot.ylabel('Sepal Width')
pyplot.title('Scatter Plot Sepal Length vs Sepal Width')
pyplot.show()

X = df.values[:, :2]
s = df['species']
d = dict([(y, x) for x, y in enumerate(sorted(set(s)))])
y = [d[x] for x in s]

clf = svm.SVC()
clf.fit(X, y)

p = clf.predict([[5.4, 3.2]])
print(p)