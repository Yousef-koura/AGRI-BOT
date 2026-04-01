import cv2
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import time
from tensorflow.keras.preprocessing import image


# Load a model
#yolov8 for leaf detection
yolov8 = YOLO(r"D:\download 99\Graduation Project\Ai model\Yolo8_model\runs\detect\train2\weights\best.pt")
# MobileNetV2 for classification
mobilenet = tf.keras.models.load_model(r"D:\download 99\Graduation Project\Ai model\mobiltnetv2_model_91%\mobilenetv2_trained_model_91%")

class_names = ['Potato Early blight',
 'Potato Healthy',
 'Potato Late blight',
 'Tomato Bacterial spot',
 'Tomato Early blight',
 'Tomato Healthy',
 'Tomato Late blight',
 'Tomato Leaf Mold',
 'Tomato Mosaic virus',
 'Tomato Septoria leaf spot',
 'Tomato Spider mites',
 'Tomato Target Spot',
 'Tomato Yellow Leaf Curl Virus'] # all 13 MobileNetV2 classes

# Define pipeline functions
def crop_object(img, box):
    x1, y1, x2, y2 = box
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    return crop

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.
    img = np.expand_dims(img, axis=0)
    return img


# Loading the image
# image_path = r"D:\download 99\Graduation Project\Ai model\mobilenetv2_model_86%\plant_disease_dataset\test\Potato Early blight\PotatoEarlyBlight(3510).jpg"
# image_path = r"D:\download 99\Graduation Project\Ai model\mobilenetv2_model_86%\plant_disease_dataset\test\Potato Late blight\PotatoLateBlight(3473).jpg"
image_path = r"D:\download 99\Graduation Project\Ai model\mobilenetv2_model_86%\plant_disease_dataset\train\Potato Early blight\PotatoEarlyBlight(3528).jpg"
# image_path = r"D:\download 99\Graduation Project\Ai model\mobilenetv2_model_86%\plant_disease_dataset\train\Potato Early blight\PotatoEarlyBlight(1441).jpg"
img = cv2.imread(image_path)

# showing the results of yolov8 model
results = yolov8(img,conf=0.5)
for r in results:
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()


boxes = results[0].boxes.xyxy.tolist()

for i, box in enumerate(boxes):
    crop = crop_object(img, box)
    cv2.imwrite(f'crop{i}.jpg', crop)
    crop = preprocess(crop)
    x = image.img_to_array(crop[0])  # Remove the extra dimension
    x = np.expand_dims(x, axis=0)

    custom = mobilenet.predict(x)
    plt.imshow(crop[0])  # Pass the first image from crop
    plt.show()

    a = custom[0]
    ind = np.argmax(a)
    confidence = a[ind] * 100  # Confidence level in percentage

    print('Prediction:', class_names[ind], f'(Confidence: {confidence:.2f}%)')

