import matplotlib.pyplot as plt
from scipy import stats

x = [0, 1, 2, 3, 4, 5, 6, 7]
y = [3, 5, 5, 6, 7, 8, 9, 11]

slope, intercept, r, p, std_err = stats.linregress(x, y)
print("slope: ", slope)
print("intercept: ", intercept)

def myfunc(x):
    return slope * x + intercept

mymodel = list(map(myfunc, x))

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear Regression Model")
plt.legend(["Best-fit Line", "Data Points"])
plt.grid(True)
plt.show()