# EXERCISE 4.4 LeNet-5 Keras Modified (Diperbarui untuk Keras versi baru)
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential()

# Memisahkan Input shape sesuai rekomendasi Keras versi terbaru
model.add(Input(shape=(32, 32, 1)))

# Modifikasi Exercise: mengubah filter menjadi 12
model.add(Conv2D(filters=12, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Perbaikan error pool_size

# Modifikasi Exercise: mengubah filter menjadi 32
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Perbaikan error pool_size

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()