import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.callbacks import Callback, ModelCheckpoint
import numpy as np

# Set path 
data_dir = 'dataset'

# input shape of the images
input_shape = (224, 224, 3)

# class
num_classes = 4
catagory = ['B', 'D', 'R', 'S']

# data generator for preprocessing data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)


# training data
train_data = datagen.flow_from_directory(
    data_dir,
    shuffle=True,
    target_size=input_shape[:2],
    batch_size=25,
    class_mode='categorical',
    subset='training')

# validation data
val_data = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=25,
    class_mode='categorical',
    subset='validation')


model = Sequential([
  layers.Conv2D(15, (7, 7), activation='relu', input_shape=input_shape),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(30, (3, 3), activation='relu'),
  layers.BatchNormalization(axis=1) ,
  layers.Dense(50, activation='relu'),
  layers.Dropout(0.3),
  layers.MaxPooling2D((2, 2)),
  layers.Conv2D(30, (3, 3), activation='relu'),
  layers.BatchNormalization(axis=1) ,
  layers.Dense(90, activation='relu'),
  layers.Dropout(0.3),
  layers.MaxPooling2D((2, 2)),
  layers.Flatten(),
  layers.Dense(115, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes, activation='softmax')
])

# Compile 
model.compile(optimizer=Adam(lr = 1e-4), loss='categorical_crossentropy', metrics=['accuracy'])


# testing data
test_data = datagen.flow_from_directory(
    data_dir,
    target_size=input_shape[:2],
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False)



class PlotLosses(Callback):
    def on_train_begin(self, logs={}):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(epoch)
        self.losses.append(logs.get('mean_absolute_error'))
        self.val_losses.append(logs.get('val_mean_absolute_error'))

        # plt.clf()
        # plt.plot(self.x, self.losses, label='mean_absolute_error')
        # plt.plot(self.x, self.val_losses, label='val_mean_absolute_error')
        # plt.legend()
        # plt.pause(0.01)


checkpoint = ModelCheckpoint('model3.h5', verbose=1, monitor='val_accuracy',save_best_only=True, mode='max')
plot_losses = PlotLosses()

#Train Model
model.fit_generator(
    train_data, 
    epochs=30, 
    validation_data=val_data, 
    validation_steps= len(val_data),
    callbacks=[checkpoint, plot_losses]
    )


#Test Model
model = load_model('model3.h5')
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

# Get the true classes
true_classes = test_data.classes

# Get the filenames
filenames = test_data.filenames


f = open("demofile2.txt", "w")

# Print the predicted class and true class for each image
for i in range(len(filenames)):
    f.write(str(filenames[i]))
    # f.write('True class:' + str(catagory[true_classes[i]]))
    f.write('::'+ str(catagory[pred_classes[i]])+'\n')
print('Testing accuracy:', test_acc)


f.close()
