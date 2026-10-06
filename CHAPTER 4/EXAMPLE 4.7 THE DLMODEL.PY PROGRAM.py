# Example 4.7 The DLmodel.py Program
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten # Updated to use Input
from keras.layers import Conv2D, MaxPooling2D

# create model
model = Sequential()

# Updated: Menggunakan Input layer untuk Keras versi baru
model.add(Input(shape=(28, 28, 1)))

# Lapisan Konvolusi
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # Ditambahkan pool_size
model.add(Dropout(0.2))

# Lapisan Fully Connected
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax')) # Misalkan klasifikasi 2 kelas

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())