# EXERCISE 4.2 Multiple Layer Perceptron dengan 3 Input
from sklearn.neural_network import MLPClassifier

# Memodifikasi variabel X agar setiap sampel memiliki 3 fitur/input
X = [[0., 0., 0.], 
     [1., 1., 1.], 
     [0., 1., 0.], 
     [1., 0., 1.]]

# Label target y
y = [0, 1, 1, 1] 

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X, y)

# Memodifikasi data yang akan diprediksi menjadi 3 input
print(clf.predict([[2., 2., 2.], [-1., -2., 0.]]))

# Mencetak ukuran bobot (koefisien) dari model yang telah dilatih
print([coef.shape for coef in clf.coefs_])