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

# Set path 
# data_dir = 'test'
data_dir = 'dataset'

# input shape of the images
input_shape = (224, 224, 3)

# class
num_classes = 4
catagory = ['B', 'D', 'R', 'S']

# data generator for preprocessing data
datagen = ImageDataGenerator(rescale=1./255)

# testing data
test_data = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=25,
    class_mode='categorical',
    # subset='validation',
    shuffle=False)

#Test Model
model = load_model('model/model3.h5')

score = model.evaluate_generator(
    test_data,
    steps=len(test_data))
print('score (mse, mae):\n',score)

test_data.reset()
predict = model.predict_generator(
    test_data,
    steps=len(test_data),
    workers = 1,
    use_multiprocessing=False)
print('prediction:\n',predict)



# Evaluate model on testing data
test_loss, test_acc = model.evaluate(test_data)

# Predict classes of testing data
pred_probs = model.predict(test_data)
pred_classes = np.argmax(pred_probs, axis=1)


# Get the filenames
filenames = test_data.filenames
# filenames = 'test'



f = open("frtest.txt", "w")

# Print the predicted class and true class for each image
for i in range(len(filenames)):
    f.write(str(filenames[i]))
    f.write('::'+ str(catagory[pred_classes[i]])+'\n')
print('Testing accuracy:', test_acc)


f.close()