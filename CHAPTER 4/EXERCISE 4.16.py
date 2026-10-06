from matplotlib import pyplot
from keras.applications.vgg19 import VGG19

# Perbaikan memori
model = VGG19(include_top=False)

n = 1
filters, biases = model.layers[n].get_weights()
s = filters.shape

f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)

# EXERCISE 4.16: Menampilkan semua filter dengan mengubah n_filters menjadi s[3]
n_filters, ix = s[3], 1

pyplot.figure(figsize=(15, 20))
for i in range(n_filters):
    f = filters[:, :, :, i]
    for j in range(s[0]):
        ax = pyplot.subplot(n_filters, s[0], ix)
        ax.set_xticks([])
        ax.set_yticks([])
        pyplot.imshow(f[:, :, j], cmap='gray')
        ix += 1
pyplot.show()