# EXERCISE 4.10 Klasifikasi Gambar dengan VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19 # Ubah import ke vgg19
from keras.applications.vgg19 import preprocess_input # Ubah import ke vgg19
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Memuat model VGG19 secara utuh
model = VGG19(weights='imagenet')

img_path = "C:/Users/elsah/OneDrive/Documents/SEMESTER 5/AI/CHAPTER 4/elephant.jpg" 
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# prediction
predictions = model.predict(x)
results = decode_predictions(predictions)

print(results)