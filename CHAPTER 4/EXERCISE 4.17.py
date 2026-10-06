from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from numpy import expand_dims
import matplotlib.pyplot as plt

# Perbaikan memori
model = VGG19(include_top=False)

# EXERCISE 4.17: Mengubah n = 1 menjadi n = 2 untuk melihat feature maps layer ke-2
n = 2 
model = Model(inputs=model.inputs, outputs=model.layers[n].output)

img_path = "C:/Users/elsah/OneDrive/Documents/SEMESTER 5/AI/CHAPTER 4/elephant.jpg"
img = load_img(img_path, target_size=(224, 224))
img = expand_dims(img, axis=0)
img = preprocess_input(img)

feature_maps = model.predict(img)

col = 8
row = int(feature_maps.shape[3]/col)
ix = 1
plt.figure(figsize=(20,20))
for _ in range(row):
    for _ in range(col):
        ax = plt.subplot(row, col, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
        ix += 1
plt.show()