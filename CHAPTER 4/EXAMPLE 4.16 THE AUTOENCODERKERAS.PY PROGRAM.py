# EXAMPLE 4.16 THE AUTOENCODERKERAS.PY PROGRAM (Disesuaikan)
import matplotlib.pyplot as plt
import numpy as np

# Menghilangkan awalan 'tensorflow.' agar linter VS Code tidak memunculkan garis kuning
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Input, InputLayer, Reshape
from keras.models import Model, Sequential

# 1. Memuat dan Normalisasi Dataset MNIST
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("Shape x_train:", x_train.shape)
print("Shape x_test:", x_test.shape)

# 2. Fungsi untuk Membangun Model AutoEncoder
def build_autoencoder(img_shape, code_size):
    # Bagian Encoder
    encoder = Sequential()
    # Perbaikan warning: input_shape diubah menjadi shape
    encoder.add(InputLayer(shape=img_shape)) 
    encoder.add(Flatten())
    encoder.add(Dense(code_size, activation="relu"))

    # Bagian Decoder
    decoder = Sequential()
    # Perbaikan warning: input_shape diubah menjadi shape
    decoder.add(InputLayer(shape=(code_size,)))
    # Perbaikan error: np.prod dibungkus dengan int()
    decoder.add(Dense(int(np.prod(img_shape)), activation="sigmoid"))
    decoder.add(Reshape(img_shape))

    return encoder, decoder

# Menentukan ukuran bentuk gambar dan ukuran kode terkompresi
IMG_SHAPE = x_train[0].shape  # Ukuran (28, 28)
CODE_SIZE = 32

encoder, decoder = build_autoencoder(IMG_SHAPE, CODE_SIZE)

# Menggabungkan Encoder dan Decoder menjadi satu model AutoEncoder
inp = Input(shape=IMG_SHAPE)
code = encoder(inp)
reconstruction = decoder(code)

autoencoder = Model(inp, reconstruction)
autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

print(autoencoder.summary())

# 3. Melatih Model AutoEncoder
history = autoencoder.fit(
    x_train,
    x_train,
    epochs=20,
    batch_size=256,
    validation_data=(x_test, x_test),
)

# 4. Visualisasi Hasil Pelatihan (Loss)
plt.figure()
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(loc="upper left")
plt.show()

# 5. Fungsi untuk Menampilkan Gambar Asli, Kode, dan Hasil Rekonstruksi
def show_image(x):
    plt.imshow(np.clip(x, 0, 1))

def visualize(img, encoder, decoder):
    """Menampilkan gambar asli, hasil encode, dan hasil decode"""
    code = encoder.predict(np.expand_dims(img, 0))[0]
    reconst = decoder.predict(np.expand_dims(code, 0))[0]

    plt.figure(figsize=(10, 3))
    
    plt.subplot(1, 3, 1)
    plt.title("Original")
    show_image(img)

    plt.subplot(1, 3, 2)
    plt.title("Code")
    show_image(code.reshape(code.shape[0] // 2, 2))

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed")
    show_image(reconst)

    plt.show()

# Menguji visualisasi pada beberapa data uji
for i in range(5):
    img = x_test[i]
    visualize(img, encoder, decoder)