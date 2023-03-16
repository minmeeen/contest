import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np
from keras.optimizers import Adam
import os
import cv2
import numpy as np

# Set path to the text file containing the image filenames
file_list_path = 'frtest.txt'

# Set path to the folder containing the images
data_dir = 'test'

# input shape of the images
input_shape = (224, 224, 3)

# class
num_classes = 4
category = ['B', 'D', 'R', 'S']

# Read in the image filenames from the text file
with open(file_list_path, 'r') as f:
    file_list = f.read().splitlines()

# Create a list to store the preprocessed images
images = []

# Loop over the filenames in the list and preprocess each image
for file_name in file_list:
    # Load the image using OpenCV
    image = cv2.imread(os.path.join(data_dir, file_name))
    # Resize the image to the input shape expected by the model
    image = cv2.resize(image, input_shape[:2])
    # Rescale the pixel values to be between 0 and 1
    image = image / 255.0
    # Add the preprocessed image to the list
    images.append(image)

# Convert the list of images to a NumPy array
images = np.array(images)

# Load the saved model
model = load_model('model/model4.h5')

# Make predictions on the preprocessed images
pred_probs = model.predict(images)
pred_classes = np.argmax(pred_probs, axis=1)
# test_loss, test_acc = model.evaluate(images)


# Write the predictions to a file
with open("testfile.txt", "w") as f:
    for i in range(len(pred_classes)):
        f.write(str(file_list[i]))
        f.write('::'+ str(category[pred_classes[i]])+'\n')
# # Write the predictions to a file
# with open("frtest.txt", "w") as f:
#     for i in range(len(pred_classes)):
#         f.write(str(os.listdir(data_dir)[i])+'\n')
#         f.write('::'+ str(category[pred_classes[i]])+'\n')



# Print the predictions
print('Predictions:', pred_classes)
# print('Testing accuracy:', test_acc)
