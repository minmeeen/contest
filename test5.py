from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2


BATCH_SIZE = 5
IMAGE_SIZE = (224,224)

dataset_col = ['filename']
dataframe = pd.read_csv('filelist.txt', names=dataset_col)

num_classes = 4
category = ['B', 'D', 'R', 'S']

datagen_noaug = ImageDataGenerator(rescale=1./255)


test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe,
    directory='test images',
    x_col='filename',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode=None
)

model = load_model('model/model4.h5')

score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)

test_generator.reset()

pred_probs = model.predict_generator(
    test_generator,
    steps=len(test_generator)
)

pred_labels = np.argmax(pred_probs, axis=1)
pred_class_names = [category[label] for label in pred_labels]


f = open("result.txt", "w")
# Print the predicted class labels for each image
for i in range(len(dataframe)):
    f.write(f'{dataframe.iloc[i]["filename"]}::{pred_class_names[i]}\n')

# print('prediction:\n', pred_probs)
