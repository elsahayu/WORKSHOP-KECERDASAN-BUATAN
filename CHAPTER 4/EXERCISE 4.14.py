# EXERCISE 4.14 Perbaikan Modul (Menggunakan Native Keras)
import cv2
import numpy as np
from keras.applications.imagenet_utils import decode_predictions

# Tentukan model yang ingin dipanggil di sini:
# Pilihan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet121', 'inceptionv3'
model_name = 'mobilenetv2'

# Menggunakan if-else untuk memuat model bawaan Keras yang kompatibel
if model_name == 'vgg16':
    from keras.applications.vgg16 import VGG16, preprocess_input
    model = VGG16(weights='imagenet')
    sz = 224
elif model_name == 'resnet50':
    from keras.applications.resnet50 import ResNet50, preprocess_input
    model = ResNet50(weights='imagenet')
    sz = 224
elif model_name == 'mobilenetv2':
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    model = MobileNetV2(weights='imagenet')
    sz = 224
elif model_name == 'densenet121':
    from keras.applications.densenet import DenseNet121, preprocess_input
    model = DenseNet121(weights='imagenet')
    sz = 224
elif model_name == 'inceptionv3':
    from keras.applications.inception_v3 import InceptionV3, preprocess_input
    model = InceptionV3(weights='imagenet')
    sz = 299 # InceptionV3 wajib menggunakan ukuran frame 299x299
else:
    print("Model tidak dikenali. Menggunakan default MobileNetV2.")
    from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
    model = MobileNetV2(weights='imagenet')
    sz = 224

print(f"Memulai kamera dengan model {model_name.upper()}...")

camera = cv2.VideoCapture(0)
while True:
    ret, cam_frame = camera.read()
    if not ret:
        break
        
    frame = cv2.resize(cam_frame, (sz, sz))
    image = np.asarray(frame)
    image = np.expand_dims(image, 0)
    
    # Menggunakan fungsi preprocess_input bawaan dari masing-masing model
    image = preprocess_input(image.copy())
    
    preds = model.predict(image)
    label = decode_predictions(preds)
    
    cv2.putText(cam_frame, "{} {:.2f}".format(label[0][0][1], label[0][0][2]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow(f"Classification - {model_name.upper()}", cam_frame)
    
    key = cv2.waitKey(30)
    if key == 27: # Tekan 'ESC' untuk keluar
        break
        
camera.release()
cv2.destroyAllWindows()


# # EXERCISE 4.14 Seleksi Model dengan if-else
# from keras.applications.imagenet_utils import decode_predictions
# from classification_models.keras import Classifiers
# import numpy as np
# import cv2

# # Tentukan model yang ingin dipanggil di sini:
# # Pilihan: 'vgg16', 'resnet50', 'mobilenetv2', 'densenet201', 'inceptionv3'
# model_name = 'mobilenetv2' 

# # Blok percabangan if-else sesuai instruksi Exercise 4.14
# if model_name == 'vgg16':
#     clf, preprocess_input = Classifiers.get('vgg16')
#     sz = 224
# elif model_name == 'resnet50':
#     clf, preprocess_input = Classifiers.get('resnet50')
#     sz = 224
# elif model_name == 'mobilenetv2':
#     clf, preprocess_input = Classifiers.get('mobilenetv2')
#     sz = 224
# elif model_name == 'densenet201':
#     clf, preprocess_input = Classifiers.get('densenet201')
#     sz = 224
# elif model_name == 'inceptionv3':
#     clf, preprocess_input = Classifiers.get('inceptionv3')
#     sz = 299 # InceptionV3 mewajibkan ukuran 299x299
# else:
#     print("Model tidak dikenali. Menggunakan default MobileNetV2.")
#     clf, preprocess_input = Classifiers.get('mobilenetv2')
#     sz = 224

# print(f"Membangun model {model_name.upper()}...")
# model = clf(input_shape=(sz, sz, 3), weights='imagenet', classes=1000)

# camera = cv2.VideoCapture(0)
# while True:
#     ret, cam_frame = camera.read()
#     if not ret:
#         break
        
#     frame = cv2.resize(cam_frame, (sz, sz))
#     image = np.asarray(frame)
#     image = np.expand_dims(image, 0)
#     image = preprocess_input(image)
    
#     preds = model.predict(image)
#     label = decode_predictions(preds)
    
#     cv2.putText(cam_frame, "{} {:.2f}".format(label[0][0][1], label[0][0][2]),
#                 (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#     cv2.imshow(f"Classification - {model_name.upper()}", cam_frame)
    
#     key = cv2.waitKey(30)
#     if key == 27: 
#         break
        
# camera.release()
# cv2.destroyAllWindows()