from matplotlib import pyplot
from keras.applications.vgg19 import VGG19

# Perbaikan memori
model = VGG19(include_top=False)

# EXERCISE 4.15: Mengubah n = 1 menjadi n = 2 untuk melihat filter layer 2
n = 2
filters, biases = model.layers[n].get_weights()
s = filters.shape

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# Menampilkan 4 filter pertama dari layer ke-2
n_filters, ix = 4, 1
pyplot.figure(figsize=(10,10))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.show()