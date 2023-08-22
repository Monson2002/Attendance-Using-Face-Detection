from cv2 import VideoCapture
import cv2
import os
import random
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

import uuid

# Putting limits to GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')
ANC_PATH = os.path.join('data', 'anchor')

# os.makedirs(POS_PATH, exist_ok=True)
# os.makedirs(NEG_PATH, exist_ok=True)
# os.makedirs(ANC_PATH, exist_ok=True)

# print(POS_PATH, NEG_PATH, ANC_PATH)

# for folder in os.listdir('lfw'):
#     for file in os.listdir(os.path.join('lfw', folder)):
#         OLD_PATH = os.path.join('lfw',folder,file)
#         NEW_PATH = os.path.join(NEG_PATH, file)
#         os.replace(OLD_PATH, NEW_PATH)
# print('Done')

# Capturing images from webcam using Open CV
cap = VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while cap.isOpened():
    ret, frame = cap.read()
    
    # Setting frame size
    frame = frame[120:120+250, 200:200+250, :]
    
    # Filling positive images
    if cv2.waitKey(1) == ord('p'):
        image = os.path.join(POS_PATH, str(uuid.uuid1()) + '.jpg')
        cv2.imwrite(image, frame)
        
    # Filling anchor images
    if cv2.waitKey(1) == ord('a'):
        image = os.path.join(ANC_PATH, str(uuid.uuid1()) + '.jpg')
        cv2.imwrite(image, frame)
        
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# Release the camera
cap.release()
cv2.destroyAllWindows()

print(plt.imshow(frame))