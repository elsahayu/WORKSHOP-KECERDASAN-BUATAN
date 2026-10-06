# Example 4.9 Simple CNN for the MNIST Dataset (Diperbarui)
from keras.datasets import mnist
from keras.models import Sequential
# 1. Perbaikan import: Gabungkan semua layer dari keras.layers
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# 2. Perbaikan import np_utils menjadi to_categorical
from keras.utils import to_categorical 

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][width][height][channels]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# one hot encode outputs (Menggunakan to_categorical)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
# 3. Perbaikan layer Input agar tidak memunculkan warning
model.add(Input(shape=(28, 28, 1))) 
model.add(Conv2D(32, (5, 5), activation='relu'))
# 4. Menambahkan pool_size secara eksplisit
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model (Catatan: Proses ini akan memakan waktu lumayan lama karena berjalan 10 epochs)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))