# Example 4.4 LeNet-5 Keras (Diperbarui)
from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten, AveragePooling2D
from keras import optimizers

model = Sequential()

# Memisahkan Input shape sesuai rekomendasi Keras versi terbaru
model.add(Input(shape=(32, 32, 1)))

# Lapisan Konvolusi dan Pooling ke-1
model.add(Conv2D(filters=6, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Menambahkan pool_size

# Lapisan Konvolusi dan Pooling ke-2
model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu'))
model.add(AveragePooling2D(pool_size=(2, 2))) # Menambahkan pool_size

model.add(Flatten())
model.add(Dense(units=120, activation='relu'))
model.add(Dense(units=84, activation='relu'))
model.add(Dense(units=10, activation='softmax'))

model.summary()