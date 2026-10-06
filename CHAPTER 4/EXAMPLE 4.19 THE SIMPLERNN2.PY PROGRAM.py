# Example 4.19 SimpleRNN2.py
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Activation
from keras import optimizers

model = Sequential()
model.add(SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
model.add(Dense(46))
model.add(Activation('softmax'))

# Perbaikan: 'lr' diubah menjadi 'learning_rate' untuk versi Keras terbaru
adam = optimizers.Adam(learning_rate=0.001)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())