# EXERCISE 4.5 AlexNet Modifikasi (Mengubah jumlah unit pada Dense Layer)
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()
model.add(Input(shape=(224, 224, 3)))

# (Lapisan konvolusi tetap sama seperti aslinya)
model.add(Conv2D(filters=96, kernel_size=(11,11), activation='relu', strides=(4,4), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(11,11), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(Conv2D(filters=384, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))

model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu', strides=(1,1), padding='valid'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

model.add(Flatten())

# ----------------- MODIFIKASI DENSE LAYER -----------------
# 1st Fully Connected Layer: Ubah dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# 2nd Fully Connected Layer: Ubah dari 4096 menjadi 2048
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.4))

# 3rd Fully Connected Layer: Ubah dari 1000 menjadi 500
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
# ----------------------------------------------------------

# Output Layer
model.add(Dense(17, activation='softmax'))

model.summary()