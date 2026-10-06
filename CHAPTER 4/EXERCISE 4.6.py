# EXERCISE 4.6 DLModel Modifikasi (Menambah layer Conv2D tambahan)
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

# create model
model = Sequential()
model.add(Input(shape=(28, 28, 1)))

# Set Lapisan Pertama (Asli)
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# ----------------- MODIFIKASI -----------------
# Set Lapisan Kedua (Tambahan sesuai Exercise)
# Menambahkan 16 filter dengan ukuran kernel 3x3
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
# ----------------------------------------------

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())