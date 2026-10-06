# Example 4.18 SimpleRNN.py
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/SimpleRNN
import tensorflow as tf
import numpy as np

i = 4
s = 5
n = 1
# Membangkitkan data input acak dengan dimensi [4, 5, 1]
inputs = np.random.random([i, s, n]).astype(np.float32)

# Menginisialisasi lapisan SimpleRNN dengan 4 unit
simple_rnn = tf.keras.layers.SimpleRNN(i)

print("Inputs:")
print(inputs)

output = simple_rnn(inputs) 
print("\nOutput:")
print(output)