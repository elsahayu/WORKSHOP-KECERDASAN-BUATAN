# Example 4.12
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
import numpy as np

# Perbaikan import agar ringkas
from keras.applications import vgg16, resnet50, mobilenet, inception_v3

# init the models (Hanya mengaktifkan VGG16 sebagai contoh agar RAM tidak penuh)
model = vgg16.VGG16(weights='imagenet')
# model = resnet50.ResNet50(weights='imagenet')
# model = mobilenet.MobileNet(weights='imagenet')
# model = inception_v3.InceptionV3(weights='imagenet')

print(model.summary())

# Pastikan path gambar elephant.jpg sudah sesuai
img_path = "C:/Users/elsah/OneDrive/Documents/SEMESTER 5/AI/CHAPTER 4/elephant.jpg"
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Preprocessing spesifik untuk masing-model
processed_image = vgg16.preprocess_input(x)
# processed_image = resnet50.preprocess_input(x)
# processed_image = mobilenet.preprocess_input(x)
# processed_image = inception_v3.preprocess_input(x)

# prediction
predictions = model.predict(processed_image)
results = decode_predictions(predictions)
print(results)