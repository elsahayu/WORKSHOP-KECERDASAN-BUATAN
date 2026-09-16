from sklearn import linear_model
from sklearn.datasets import load_linnerud

# Memuat dataset linnerud
X, y = load_linnerud(return_X_y=True)

# Membuat dan melatih model
reg = linear_model.LinearRegression()
reg.fit(X, y)

# Menampilkan koefisien dan intersep
print('Coefficients: \n', reg.coef_)
print('Intercept: \n', reg.intercept_)

# Melakukan prediksi menggunakan baris pertama data sebagai contoh
pred = reg.predict([X[0]])
print('Prediction: \n', pred)