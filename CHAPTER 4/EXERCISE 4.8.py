# EXERCISE 4.8 Penambahan Layer Ekstra (Bebas Error Keras Baru)
from keras.datasets import mnist
from keras.models import Sequential
# Perbaikan import layer dan to_categorical
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshape & normalize
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255

# One hot encode
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
num_classes = y_test.shape[1]

# Create model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))

# Layer Set Pertama (Asli dari Example 4.9)
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# MODIFIKASI: Menambahkan Layer Set Kedua sebelum Flatten
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Opsional: Jika ingin melatih modelnya, uncomment baris di bawah ini
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)