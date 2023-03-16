from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np
import cv2


BATCH_SIZE = 5
IMAGE_SIZE = (224,224)

#Download dataset form https://drive.google.com/file/d/1jwa16s2nZIQywKMdRkpRvdDifxGDxC3I/view?usp=sharing
dataset_col = ['filename']
dataframe = pd.read_csv('frtest.txt', names=dataset_col)
# print(dataframe.iloc[0])


num_classes = 4
category = ['B', 'D', 'R', 'S']

datagen_noaug = ImageDataGenerator(rescale=1./255)


test_generator = datagen_noaug.flow_from_dataframe(
    dataframe=dataframe,
    directory='test',
    x_col='filename',
    # y_col='norm_weight',
    shuffle=False,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
    # class_mode=None
    )

model = load_model('model/model4.h5')

score = model.evaluate_generator(
    test_generator,
    steps=len(test_generator))
print('score (mse, mae):\n',score)




test_generator.reset()
# predict = model.predict_generator(
#     test_generator,
#     steps=len(test_generator),
#     workers = 1,
#     use_multiprocessing=False)

pred_probs = model.predict_generator(
    test_generator,
    steps=len(test_generator))

pred_labels = np.argmax(pred_probs, axis=1)
category = ['B', 'D', 'R', 'S']
pred_class_names = [category[label] for label in pred_labels]

# Print the predicted class labels for each image
for i in range(len(dataframe)):
    print(f'{dataframe.iloc[i]["filename"]}: {pred_class_names[i]}')

print('prediction:\n',predict)


# imgfile = 'rice/images/001_t.bmp'
# test_im = cv2.imread(imgfile, cv2.IMREAD_COLOR)
# test_im = cv2.cvtColor(test_im, cv2.COLOR_BGR2RGB)
# test_im = cv2.resize(test_im, IMAGE_SIZE)
# test_im = test_im / 255.
# test_im = np.expand_dims(test_im, axis=0)
# w_pred = model.predict(test_im)
# print(imgfile, " = ", w_pred[0][0])